from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from airgap_rag.db.models.documents import Document
from airgap_rag.db.models.jobs import Job
from airgap_rag.db.repositories.jobs import ClaimedJob
from airgap_rag.documents.types import DocumentStatus
from airgap_rag.indexing.service import DocumentIndexingError, IndexingResult, IndexingService
from airgap_rag.jobs.errors import RetryableJobError
from airgap_rag.jobs.service import JobExecutionOutcome, JobExecutionService
from airgap_rag.jobs.state_machine import reset_job_for_retry, transition_job
from airgap_rag.jobs.types import JobStatus, JobType


class FakeJobRepository:
    def __init__(self, job: Job, document: Document, *, claimable: bool = True) -> None:
        self.job = job
        self.document = document
        self.claimable = claimable
        self.rollbacks = 0

    async def get(self, job_id: UUID) -> Job | None:
        return self.job if self.job.id == job_id else None

    async def claim(self, job_id: UUID, *, stale_after_seconds: int) -> ClaimedJob | None:
        del stale_after_seconds
        if not self.claimable or job_id != self.job.id:
            return None
        transition_job(self.job, JobStatus.PARSING)
        self.job.attempt += 1
        return ClaimedJob(self.job, self.document)

    async def prepare_retry(
        self,
        job_id: UUID,
        *,
        error_code: str,
        error_message: str,
    ) -> None:
        assert job_id == self.job.id
        reset_job_for_retry(self.job)
        self.job.error_code = error_code
        self.job.error_message = error_message
        self.document.status = DocumentStatus.PENDING

    async def fail(
        self,
        job_id: UUID,
        *,
        error_code: str,
        error_message: str,
    ) -> None:
        assert job_id == self.job.id
        transition_job(self.job, JobStatus.FAILED)
        self.job.error_code = error_code
        self.job.error_message = error_message
        self.document.status = DocumentStatus.FAILED

    async def rollback(self) -> None:
        self.rollbacks += 1


class StubIndexingService(IndexingService):
    def __init__(self, job: Job, document: Document, error: Exception | None = None) -> None:
        self.job = job
        self.document = document
        self.error = error

    async def index_document(
        self,
        document_id: UUID,
        *,
        mark_failed_on_error: bool = True,
    ) -> IndexingResult:
        assert mark_failed_on_error is False
        if self.error is not None:
            raise self.error
        for status in (
            JobStatus.CHUNKING,
            JobStatus.EMBEDDING,
            JobStatus.INDEXING,
            JobStatus.READY,
        ):
            transition_job(self.job, status)
        self.document.status = DocumentStatus.READY
        return IndexingResult(document_id, 1, DocumentStatus.READY)


def make_models(*, attempt: int = 0) -> tuple[Job, Document]:
    now = datetime.now(UTC)
    document = Document(
        id=uuid4(),
        filename="notes.txt",
        storage_key="documents/id/notes.txt",
        mime_type="text/plain",
        size=5,
        sha256="a" * 64,
        status=DocumentStatus.PENDING,
        page_count=None,
        created_at=now,
        updated_at=now,
    )
    job = Job(
        id=uuid4(),
        document_id=document.id,
        type=JobType.INGESTION,
        status=JobStatus.PENDING,
        progress=0,
        attempt=attempt,
        created_at=now,
        updated_at=now,
    )
    return job, document


def make_service(
    repository: FakeJobRepository,
    indexing_service: StubIndexingService,
) -> JobExecutionService:
    return JobExecutionService(
        repository,
        indexing_service,
        max_attempts=3,
        timeout_seconds=1,
        stale_after_seconds=2,
    )


async def test_successful_job_reaches_ready() -> None:
    job, document = make_models()
    repository = FakeJobRepository(job, document)

    outcome = await make_service(repository, StubIndexingService(job, document)).execute(job.id)

    assert outcome == JobExecutionOutcome.COMPLETED
    assert job.status == JobStatus.READY
    assert job.attempt == 1


async def test_transient_failure_is_reset_for_taskiq_retry() -> None:
    job, document = make_models()
    repository = FakeJobRepository(job, document)

    with pytest.raises(RetryableJobError, match="qdrant unavailable"):
        await make_service(
            repository,
            StubIndexingService(job, document, ConnectionError("qdrant unavailable")),
        ).execute(job.id)

    assert job.status == JobStatus.PENDING
    assert document.status == DocumentStatus.PENDING
    assert job.error_code == "infrastructure_unavailable"
    assert repository.rollbacks == 1


async def test_permanent_failure_is_not_retried() -> None:
    job, document = make_models()
    repository = FakeJobRepository(job, document)

    outcome = await make_service(
        repository,
        StubIndexingService(job, document, DocumentIndexingError("empty document")),
    ).execute(job.id)

    assert outcome == JobExecutionOutcome.FAILED
    assert job.status == JobStatus.FAILED
    assert job.error_code == "document_indexing_failed"


async def test_transient_failure_stops_after_max_attempts() -> None:
    job, document = make_models(attempt=2)
    repository = FakeJobRepository(job, document)

    outcome = await make_service(
        repository,
        StubIndexingService(job, document, ConnectionError("still unavailable")),
    ).execute(job.id)

    assert outcome == JobExecutionOutcome.FAILED
    assert job.attempt == 3
    assert job.status == JobStatus.FAILED


async def test_duplicate_delivery_is_skipped_when_job_cannot_be_claimed() -> None:
    job, document = make_models()
    repository = FakeJobRepository(job, document, claimable=False)

    outcome = await make_service(repository, StubIndexingService(job, document)).execute(job.id)

    assert outcome == JobExecutionOutcome.SKIPPED
    assert job.attempt == 0
