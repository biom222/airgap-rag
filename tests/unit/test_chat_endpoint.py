from uuid import UUID, uuid4

from httpx import ASGITransport, AsyncClient

from airgap_rag.api.dependencies.chat import get_chat_service
from airgap_rag.core.config import Settings
from airgap_rag.embeddings.mock import MockEmbeddingProvider
from airgap_rag.main import create_app
from airgap_rag.rag.service import RAGService
from airgap_rag.rag.types import Citation, RAGAnswer
from tests.conftest import FakeDatabase, FakeJobPublisher, FakeVectorStore


class StubRAGService(RAGService):
    def __init__(self, result: RAGAnswer) -> None:
        self.result = result

    async def answer(
        self,
        question: str,
        *,
        session_id: UUID | None,
        document_ids: list[UUID] | None,
        top_k: int,
        top_n: int,
    ) -> RAGAnswer:
        del question, session_id, document_ids, top_k, top_n
        return self.result


async def test_chat_endpoint_returns_answer_session_and_sources(settings: Settings) -> None:
    session_id = uuid4()
    source = Citation(
        document_id=uuid4(),
        filename="policy.pdf",
        page=7,
        chunk_id=uuid4(),
        vector_score=0.93,
        rerank_score=0.97,
    )
    service = StubRAGService(
        RAGAnswer(answer="Five years [S1]", session_id=session_id, sources=(source,))
    )
    application = create_app(
        settings=settings,
        database=FakeDatabase(),
        embedding_provider=MockEmbeddingProvider(8),
        vector_store=FakeVectorStore(),
        job_publisher=FakeJobPublisher(),
    )
    application.dependency_overrides[get_chat_service] = lambda: service

    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application), base_url="http://test"
        ) as client:
            response = await client.post("/api/v1/chat", json={"question": "How long?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Five years [S1]",
        "session_id": str(session_id),
        "sources": [
            {
                "document_id": str(source.document_id),
                "filename": "policy.pdf",
                "page": 7,
                "chunk_id": str(source.chunk_id),
                "score": 0.93,
                "rerank_score": 0.97,
            }
        ],
    }


async def test_chat_endpoint_rejects_blank_question(settings: Settings) -> None:
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
            response = await client.post("/api/v1/chat", json={"question": "   "})

    assert response.status_code == 422
