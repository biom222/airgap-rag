from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from sqlalchemy import Float, cast, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from airgap_rag.chunking.base import Chunk
from airgap_rag.db.models.documents import Document, DocumentChunk
from airgap_rag.db.models.jobs import Job
from airgap_rag.db.protocols import AsyncSessionProvider
from airgap_rag.documents.types import DocumentStatus
from airgap_rag.jobs.state_machine import transition_job
from airgap_rag.jobs.types import JobStatus
from airgap_rag.retrieval.filters import RetrievalFilters


@dataclass(frozen=True, slots=True)
class LexicalSearchHit:
    chunk_id: UUID
    filename: str
    score: float


class IndexingRepository(Protocol):
    async def get_document(self, document_id: UUID) -> Document | None: ...

    async def set_status(
        self,
        document: Document,
        status: DocumentStatus,
        *,
        page_count: int | None = None,
    ) -> None: ...

    async def replace_chunks(self, document_id: UUID, chunks: tuple[Chunk, ...]) -> None: ...

    async def get_chunks(
        self, chunk_ids: list[UUID], owner_id: str | None = None
    ) -> dict[UUID, DocumentChunk]: ...

    async def lexical_search(
        self,
        query: str,
        *,
        limit: int,
        document_ids: list[UUID] | None = None,
        owner_id: str | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[LexicalSearchHit]: ...

    async def rollback(self) -> None: ...


class SQLAlchemyIndexingRepository:
    def __init__(self, session: AsyncSession, job_id: UUID | None = None) -> None:
        self._session = session
        self._job_id = job_id

    async def get_document(self, document_id: UUID) -> Document | None:
        return await self._session.get(Document, document_id)

    async def set_status(
        self,
        document: Document,
        status: DocumentStatus,
        *,
        page_count: int | None = None,
    ) -> None:
        document.status = status
        if page_count is not None:
            document.page_count = page_count
        if self._job_id is not None:
            job = await self._session.get(Job, self._job_id)
            if job is None:
                raise RuntimeError(f"Job {self._job_id} disappeared during indexing")
            transition_job(job, JobStatus(status.value))
        await self._session.commit()

    async def replace_chunks(self, document_id: UUID, chunks: tuple[Chunk, ...]) -> None:
        await self._session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        self._session.add_all(
            [
                DocumentChunk(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    page=chunk.page,
                    text=chunk.text,
                    text_hash=chunk.text_hash,
                    section=chunk.section,
                    content_type=chunk.content_type,
                )
                for chunk in chunks
            ]
        )
        await self._session.commit()

    async def get_chunks(
        self, chunk_ids: list[UUID], owner_id: str | None = None
    ) -> dict[UUID, DocumentChunk]:
        if not chunk_ids:
            return {}
        statement = select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
        if owner_id is not None:
            statement = statement.join(Document).where(Document.owner_id == owner_id)
        result = await self._session.execute(statement)
        return {chunk.id: chunk for chunk in result.scalars()}

    @staticmethod
    def _apply_document_filters(statement, filters: RetrievalFilters | None):  # type: ignore[no-untyped-def]
        if filters is None:
            return statement
        if filters.department is not None:
            statement = statement.where(Document.department == filters.department)
        if filters.document_type is not None:
            statement = statement.where(Document.document_type == filters.document_type)
        if filters.date_from is not None:
            statement = statement.where(Document.document_date >= filters.date_from)
        if filters.date_to is not None:
            statement = statement.where(Document.document_date <= filters.date_to)
        if filters.max_access_level is not None:
            statement = statement.where(Document.access_level <= filters.max_access_level)
        for tag in filters.tags:
            statement = statement.where(Document.tags.contains([tag]))
        return statement

    async def lexical_search(
        self,
        query: str,
        *,
        limit: int,
        document_ids: list[UUID] | None = None,
        owner_id: str | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[LexicalSearchHit]:
        ts_query = func.websearch_to_tsquery("simple", query)
        vector = func.to_tsvector("simple", DocumentChunk.text)
        rank = cast(func.ts_rank_cd(vector, ts_query), Float).label("rank")
        statement = (
            select(DocumentChunk.id, Document.filename, rank)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(vector.op("@@")(ts_query))
        )
        if document_ids:
            statement = statement.where(Document.id.in_(document_ids))
        if owner_id is not None:
            statement = statement.where(Document.owner_id == owner_id)
        statement = self._apply_document_filters(statement, filters)
        rows = (await self._session.execute(statement.order_by(rank.desc()).limit(limit))).all()
        return [
            LexicalSearchHit(chunk_id=row.id, filename=row.filename, score=float(row.rank))
            for row in rows
        ]

    async def filter_chunks(
        self,
        chunk_ids: list[UUID],
        *,
        owner_id: str | None,
        filters: RetrievalFilters | None,
    ) -> dict[UUID, DocumentChunk]:
        if not chunk_ids:
            return {}
        statement = (
            select(DocumentChunk)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.id.in_(chunk_ids))
        )
        if owner_id is not None:
            statement = statement.where(Document.owner_id == owner_id)
        statement = self._apply_document_filters(statement, filters)
        result = await self._session.execute(statement)
        return {chunk.id: chunk for chunk in result.scalars()}

    async def rollback(self) -> None:
        await self._session.rollback()


class SessionChunkRepository:
    """Loads retrieval chunks without holding a transaction during LLM inference."""

    def __init__(self, database: AsyncSessionProvider, owner_id: str | None = None) -> None:
        self._database = database
        self._owner_id = owner_id

    async def get_chunks(self, chunk_ids: list[UUID]) -> dict[UUID, DocumentChunk]:
        async with self._database.session() as session:
            return await SQLAlchemyIndexingRepository(session).get_chunks(chunk_ids, self._owner_id)

    async def lexical_search(
        self,
        query: str,
        *,
        limit: int,
        document_ids: list[UUID] | None = None,
        owner_id: str | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[LexicalSearchHit]:
        async with self._database.session() as session:
            return await SQLAlchemyIndexingRepository(session).lexical_search(
                query,
                limit=limit,
                document_ids=document_ids,
                owner_id=self._owner_id if owner_id is None else owner_id,
                filters=filters,
            )

    async def filter_chunks(
        self,
        chunk_ids: list[UUID],
        *,
        owner_id: str | None,
        filters: RetrievalFilters | None,
    ) -> dict[UUID, DocumentChunk]:
        async with self._database.session() as session:
            return await SQLAlchemyIndexingRepository(session).filter_chunks(
                chunk_ids,
                owner_id=self._owner_id if owner_id is None else owner_id,
                filters=filters,
            )
