from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

from httpx import ASGITransport, AsyncClient

from airgap_rag.api.dependencies.jobs import get_job_query_service
from airgap_rag.core.config import Settings
from airgap_rag.db.models.jobs import Job
from airgap_rag.embeddings.mock import MockEmbeddingProvider
from airgap_rag.jobs.service import JobQueryService
from airgap_rag.jobs.types import JobStatus, JobType
from airgap_rag.main import create_app
from tests.conftest import FakeDatabase, FakeJobPublisher, FakeVectorStore


class StubJobQueryService(JobQueryService):
    def __init__(self, job: Job) -> None:
        self.job = job

    async def get(self, job_id: UUID) -> Job:
        assert job_id == self.job.id
        return self.job


async def test_get_job_returns_progress_and_failure_fields(settings: Settings) -> None:
    now = datetime.now(UTC)
    job = Job(
        id=uuid4(),
        document_id=uuid4(),
        type=JobType.INGESTION,
        status=JobStatus.EMBEDDING,
        progress=50,
        attempt=2,
        error_code=None,
        error_message=None,
        created_at=now,
        started_at=now,
        finished_at=None,
        updated_at=now,
    )
    service = StubJobQueryService(job)
    application = create_app(
        settings=settings,
        database=FakeDatabase(),
        embedding_provider=MockEmbeddingProvider(8),
        vector_store=FakeVectorStore(),
        job_publisher=FakeJobPublisher(),
    )

    async def override_service() -> AsyncIterator[JobQueryService]:
        yield service

    application.dependency_overrides[get_job_query_service] = override_service
    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application), base_url="http://test"
        ) as client:
            response = await client.get(f"/api/v1/jobs/{job.id}")

    assert response.status_code == 200
    assert response.json()["status"] == "EMBEDDING"
    assert response.json()["progress"] == 50
    assert response.json()["attempt"] == 2
