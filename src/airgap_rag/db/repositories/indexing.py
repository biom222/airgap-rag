from typing import Protocol
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from airgap_rag.chunking.base import Chunk
from airgap_rag.db.models.documents import Document, DocumentChunk
from airgap_rag.db.models.jobs import Job
from airgap_rag.documents.types import DocumentStatus
from airgap_rag.jobs.state_machine import transition_job
from airgap_rag.jobs.types import JobStatus


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

    async def get_chunks(self, chunk_ids: list[UUID]) -> dict[UUID, DocumentChunk]: ...

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
                )
                for chunk in chunks
            ]
        )
        await self._session.commit()

    async def get_chunks(self, chunk_ids: list[UUID]) -> dict[UUID, DocumentChunk]:
        if not chunk_ids:
            return {}
        result = await self._session.execute(
            select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
        )
        return {chunk.id: chunk for chunk in result.scalars()}

    async def rollback(self) -> None:
        await self._session.rollback()
