from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from airgap_rag.db.models.documents import Document
from airgap_rag.db.models.jobs import Job


class DocumentRepository(Protocol):
    async def find_by_sha256(self, sha256: str) -> Document | None: ...

    async def get(self, document_id: UUID) -> Document | None: ...

    async def find_latest_job(self, document_id: UUID) -> Job | None: ...

    async def save(self, document: Document, job: Job) -> None: ...

    async def rollback(self) -> None: ...


class SQLAlchemyDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_by_sha256(self, sha256: str) -> Document | None:
        result = await self._session.execute(select(Document).where(Document.sha256 == sha256))
        return result.scalar_one_or_none()

    async def get(self, document_id: UUID) -> Document | None:
        return await self._session.get(Document, document_id)

    async def find_latest_job(self, document_id: UUID) -> Job | None:
        result = await self._session.execute(
            select(Job)
            .where(Job.document_id == document_id)
            .order_by(Job.created_at.desc(), Job.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def save(self, document: Document, job: Job) -> None:
        self._session.add_all((document, job))
        await self._session.commit()
        await self._session.refresh(document)
        await self._session.refresh(job)

    async def rollback(self) -> None:
        await self._session.rollback()
