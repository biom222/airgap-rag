from collections.abc import AsyncIterator
from uuid import UUID, uuid4

from httpx import ASGITransport, AsyncClient

from airgap_rag.api.dependencies.retrieval import get_retrieval_service
from airgap_rag.core.config import Settings
from airgap_rag.embeddings.mock import MockEmbeddingProvider
from airgap_rag.main import create_app
from airgap_rag.retrieval.service import RetrievalService, RetrievedChunk
from tests.conftest import FakeDatabase, FakeJobPublisher, FakeVectorStore


class StubRetrievalService(RetrievalService):
    def __init__(self, result: RetrievedChunk) -> None:
        self.result = result
        self.received_top_k: int | None = None

    async def search(
        self,
        question: str,
        *,
        top_k: int,
        document_ids: list[UUID] | None = None,
    ) -> list[RetrievedChunk]:
        self.received_top_k = top_k
        return [self.result]


async def test_retrieval_endpoint_returns_debug_metadata(settings: Settings) -> None:
    document_id = uuid4()
    service = StubRetrievalService(
        RetrievedChunk(
            document_id=document_id,
            filename="policy.pdf",
            page=7,
            chunk_id=uuid4(),
            chunk_index=3,
            text="Retention is five years.",
            vector_score=0.93,
        )
    )
    application = create_app(
        settings=settings,
        database=FakeDatabase(),
        embedding_provider=MockEmbeddingProvider(8),
        vector_store=FakeVectorStore(),
        job_publisher=FakeJobPublisher(),
    )

    async def override_service() -> AsyncIterator[RetrievalService]:
        yield service

    application.dependency_overrides[get_retrieval_service] = override_service
    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/retrieval/search",
                json={"question": "How long?", "document_ids": [str(document_id)], "top_k": 4},
            )

    assert response.status_code == 200
    assert response.json()["results"][0]["filename"] == "policy.pdf"
    assert response.json()["results"][0]["vector_score"] == 0.93
    assert service.received_top_k == 4


async def test_retrieval_endpoint_rejects_blank_question(settings: Settings) -> None:
    service = StubRetrievalService(
        RetrievedChunk(
            document_id=uuid4(),
            filename="unused.txt",
            page=None,
            chunk_id=uuid4(),
            chunk_index=0,
            text="unused",
            vector_score=0.0,
        )
    )
    application = create_app(
        settings=settings,
        database=FakeDatabase(),
        embedding_provider=MockEmbeddingProvider(8),
        vector_store=FakeVectorStore(),
        job_publisher=FakeJobPublisher(),
    )

    async def override_service() -> AsyncIterator[RetrievalService]:
        yield service

    application.dependency_overrides[get_retrieval_service] = override_service
    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application), base_url="http://test"
        ) as client:
            response = await client.post("/api/v1/retrieval/search", json={"question": "   "})

    assert response.status_code == 422
