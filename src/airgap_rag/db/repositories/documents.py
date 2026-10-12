from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from airgap_rag.db.models.documents import Document
from airgap_rag.db.models.jobs import Job
from airgap_rag.db.models.outbox import OutboxEvent


class DocumentRepository(Protocol):
    async def find_by_sha256(self, owner_id: str, sha256: str) -> Document | None: ...

    async def get(self, document_id: UUID, owner_id: str | None) -> Document | None: ...

    async def find_latest_job(self, document_id: UUID) -> Job | None: ...

    async def save(self, document: Document, job: Job, event: OutboxEvent) -> None: ...

    async def ensure_outbox(
        self,
        job: Job,
        *,
        trace_id: str | None,
        trace_context: dict[str, str],
    ) -> None: ...

    async def rollback(self) -> None: ...


class SQLAlchemyDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_by_sha256(self, owner_id: str, sha256: str) -> Document | None:
        result = await self._session.execute(
            select(Document).where(Document.owner_id == owner_id, Document.sha256 == sha256)
        )
        return result.scalar_one_or_none()

    async def get(self, document_id: UUID, owner_id: str | None) -> Document | None:
        statement = select(Document).where(Document.id == document_id)
        if owner_id is not None:
            statement = statement.where(Document.owner_id == owner_id)
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def find_latest_job(self, document_id: UUID) -> Job | None:
        result = await self._session.execute(
            select(Job)
            .where(Job.document_id == document_id)
            .order_by(Job.created_at.desc(), Job.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def save(self, document: Document, job: Job, event: OutboxEvent) -> None:
        self._session.add_all((document, job, event))
        await self._session.commit()
        await self._session.refresh(document)
        await self._session.refresh(job)

    async def ensure_outbox(
        self,
        job: Job,
        *,
        trace_id: str | None,
        trace_context: dict[str, str],
    ) -> None:
        result = await self._session.execute(
            select(OutboxEvent.id).where(
                OutboxEvent.aggregate_id == job.id,
                OutboxEvent.event_type == "ingestion.requested",
                OutboxEvent.processed_at.is_(None),
            )
        )
        if result.scalar_one_or_none() is not None:
            return
        self._session.add(
            OutboxEvent(
                id=uuid4(),
                event_type="ingestion.requested",
                aggregate_id=job.id,
                payload={
                    "job_id": str(job.id),
                    "trace_id": trace_id or "",
                    "trace_context": trace_context,
                },
                attempts=0,
            )
        )
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()
