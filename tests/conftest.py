import shutil
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from airgap_rag.core.config import Settings
from airgap_rag.embeddings.mock import MockEmbeddingProvider
from airgap_rag.jobs.service import JobPublisher
from airgap_rag.main import create_app
from airgap_rag.vector_store.base import VectorPoint, VectorSearchResult


class FakeDatabase:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.disposed = False

    async def ping(self) -> None:
        if self.error is not None:
            raise self.error

    async def dispose(self) -> None:
        self.disposed = True


class FakeVectorStore:
    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy
        self.closed = False

    async def ensure_collection(self, dimension: int) -> None:
        return None

    async def delete_document(self, document_id: UUID) -> None:
        return None

    async def upsert(self, points: list[VectorPoint]) -> None:
        return None

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        document_ids: list[UUID] | None = None,
    ) -> list[VectorSearchResult]:
        return []

    async def healthcheck(self) -> bool:
        return self.healthy

    async def close(self) -> None:
        self.closed = True


class FakeJobPublisher(JobPublisher):
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.enqueued: list[UUID] = []

    async def enqueue_ingestion(self, job_id: UUID) -> None:
        if self.error is not None:
            raise self.error
        self.enqueued.append(job_id)


@pytest.fixture
def runtime_path() -> Iterator[Path]:
    path = Path("tests/.runtime") / uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_env="test",
        database_url="postgresql+asyncpg://test:test@localhost:5432/test",
        llm_provider="mock",
    )


@pytest.fixture
def fake_database() -> FakeDatabase:
    return FakeDatabase()


@pytest.fixture
async def client(settings: Settings, fake_database: FakeDatabase) -> AsyncIterator[AsyncClient]:
    application = create_app(
        settings=settings,
        database=fake_database,
        embedding_provider=MockEmbeddingProvider(8),
        vector_store=FakeVectorStore(),
        job_publisher=FakeJobPublisher(),
    )
    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://test",
        ) as test_client:
            yield test_client
