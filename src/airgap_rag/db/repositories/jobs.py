from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from airgap_rag.db.models.documents import Document
from airgap_rag.db.models.jobs import Job
from airgap_rag.documents.types import DocumentStatus
from airgap_rag.jobs.state_machine import reset_job_for_retry, transition_job
from airgap_rag.jobs.types import ACTIVE_JOB_STATUSES, JobStatus


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    job: Job
    document: Document


class JobRepository(Protocol):
    async def get(self, job_id: UUID) -> Job | None: ...

    async def claim(self, job_id: UUID, *, stale_after_seconds: int) -> ClaimedJob | None: ...

    async def prepare_retry(
        self,
        job_id: UUID,
        *,
        error_code: str,
        error_message: str,
    ) -> None: ...

    async def fail(
        self,
        job_id: UUID,
        *,
        error_code: str,
        error_message: str,
    ) -> None: ...

    async def rollback(self) -> None: ...


class SQLAlchemyJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, job_id: UUID) -> Job | None:
        return await self._session.get(Job, job_id)

    async def claim(self, job_id: UUID, *, stale_after_seconds: int) -> ClaimedJob | None:
        result = await self._session.execute(select(Job).where(Job.id == job_id).with_for_update())
        job = result.scalar_one_or_none()
        if job is None or job.status in {JobStatus.READY, JobStatus.FAILED}:
            await self._session.rollback()
            return None

        now = datetime.now(UTC)
        if job.status in ACTIVE_JOB_STATUSES:
            stale_before = now - timedelta(seconds=stale_after_seconds)
            updated_at = job.updated_at
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)
            if updated_at > stale_before:
                await self._session.rollback()
                return None
            reset_job_for_retry(job)
            document = await self._session.get(Document, job.document_id)
            if document is None:
                await self._session.rollback()
                return None
            document.status = DocumentStatus.PENDING

        document = await self._session.get(Document, job.document_id)
        if document is None:
            await self._session.rollback()
            return None

        transition_job(job, JobStatus.PARSING, now=now)
        job.attempt += 1
        job.error_code = None
        job.error_message = None
        await self._session.commit()
        await self._session.refresh(job)
        return ClaimedJob(job=job, document=document)

    async def prepare_retry(
        self,
        job_id: UUID,
        *,
        error_code: str,
        error_message: str,
    ) -> None:
        job, document = await self._load_for_update(job_id)
        reset_job_for_retry(job)
        job.error_code = error_code
        job.error_message = error_message
        document.status = DocumentStatus.PENDING
        await self._session.commit()

    async def fail(
        self,
        job_id: UUID,
        *,
        error_code: str,
        error_message: str,
    ) -> None:
        job, document = await self._load_for_update(job_id)
        transition_job(job, JobStatus.FAILED)
        job.error_code = error_code
        job.error_message = error_message
        document.status = DocumentStatus.FAILED
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()

    async def _load_for_update(self, job_id: UUID) -> tuple[Job, Document]:
        result = await self._session.execute(select(Job).where(Job.id == job_id).with_for_update())
        job = result.scalar_one_or_none()
        if job is None:
            raise RuntimeError(f"Job {job_id} disappeared during failure handling")
        document = await self._session.get(Document, job.document_id)
        if document is None:
            raise RuntimeError(f"Document {job.document_id} disappeared during failure handling")
        return job, document
