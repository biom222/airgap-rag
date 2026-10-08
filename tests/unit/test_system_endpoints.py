from httpx import ASGITransport, AsyncClient

from airgap_rag.core.config import Settings
from airgap_rag.embeddings.mock import MockEmbeddingProvider
from airgap_rag.main import create_app
from tests.conftest import FakeDatabase, FakeVectorStore


async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]


async def test_health_preserves_client_request_id(client: AsyncClient) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "request-from-client"})

    assert response.headers["X-Request-ID"] == "request-from-client"


async def test_ready_returns_ready_when_database_is_available(client: AsyncClient) -> None:
    response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"database": "ok", "embeddings": "ok", "qdrant": "ok"},
    }


async def test_ready_returns_503_when_database_is_unavailable(settings: Settings) -> None:
    database = FakeDatabase(ConnectionError("database is unavailable"))
    application = create_app(
        settings=settings,
        database=database,
        embedding_provider=MockEmbeddingProvider(8),
        vector_store=FakeVectorStore(),
    )
    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://test",
        ) as test_client:
            response = await test_client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "unavailable", "embeddings": "ok", "qdrant": "ok"},
    }


async def test_ready_returns_503_when_qdrant_is_unavailable(settings: Settings) -> None:
    application = create_app(
        settings=settings,
        database=FakeDatabase(),
        embedding_provider=MockEmbeddingProvider(8),
        vector_store=FakeVectorStore(healthy=False),
    )
    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://test",
        ) as test_client:
            response = await test_client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "ok", "embeddings": "ok", "qdrant": "unavailable"},
    }
