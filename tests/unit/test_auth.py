import json

from httpx import ASGITransport, AsyncClient

from airgap_rag.auth import Authenticator, Role
from airgap_rag.core.config import Settings
from airgap_rag.embeddings.mock import MockEmbeddingProvider
from airgap_rag.main import create_app
from tests.conftest import FakeDatabase, FakeJobPublisher, FakeVectorStore


def principal_config(token: str) -> str:
    return json.dumps(
        [
            {
                "subject": "alice",
                "role": "user",
                "token_sha256": Authenticator.token_digest(token),
            }
        ]
    )


def test_authenticator_uses_hashed_bearer_tokens() -> None:
    authenticator = Authenticator(required=True, principals_json=principal_config("secret"))

    assert authenticator.authenticate(None) is None
    assert authenticator.authenticate("wrong") is None
    principal = authenticator.authenticate("secret")
    assert principal is not None
    assert principal.subject == "alice"
    assert principal.role is Role.USER


async def test_protected_endpoint_requires_valid_bearer_token() -> None:
    settings = Settings(
        app_env="test",
        database_url="postgresql+asyncpg://test:test@localhost:5432/test",
        auth_required=True,
        auth_principals_json=principal_config("secret"),
    )
    application = create_app(
        settings=settings,
        database=FakeDatabase(),
        embedding_provider=MockEmbeddingProvider(8),
        vector_store=FakeVectorStore(),
        job_publisher=FakeJobPublisher(),
    )
    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application), base_url="http://test"
        ) as client:
            missing = await client.get("/system/info")
            invalid = await client.get("/system/info", headers={"Authorization": "Bearer wrong"})
            valid = await client.get("/system/info", headers={"Authorization": "Bearer secret"})

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert valid.status_code == 200


async def test_production_disables_interactive_api_schema() -> None:
    settings = Settings(
        app_env="production",
        database_url="postgresql+asyncpg://airgap:random-password@postgres:5432/airgap_rag",
        redis_url="redis://:random-password@redis:6379/0",
        qdrant_api_key="random-api-key",
        auth_required=True,
        auth_principals_json=principal_config("secret"),
    )
    application = create_app(
        settings=settings,
        database=FakeDatabase(),
        embedding_provider=MockEmbeddingProvider(8),
        vector_store=FakeVectorStore(),
        job_publisher=FakeJobPublisher(),
    )
    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application), base_url="http://test"
        ) as client:
            docs = await client.get("/docs")
            schema = await client.get("/openapi.json")

    assert docs.status_code == 404
    assert schema.status_code == 404
