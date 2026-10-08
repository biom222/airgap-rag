from collections.abc import AsyncGenerator
from uuid import UUID, uuid4

import pytest
from fastapi import Request
from httpx import ASGITransport, AsyncClient

from airgap_rag.api.dependencies.chat import get_chat_service
from airgap_rag.core.config import Settings
from airgap_rag.embeddings.mock import MockEmbeddingProvider
from airgap_rag.main import create_app
from airgap_rag.rag.errors import ChatSessionNotFoundError, LLMServiceUnavailableError
from airgap_rag.rag.service import RAGService
from airgap_rag.rag.types import (
    Citation,
    RAGAnswer,
    RAGStream,
    RAGStreamDone,
    RAGStreamEvent,
    RAGStreamToken,
)
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


class StubStreamingRAGService(RAGService):
    def __init__(
        self,
        sources: tuple[Citation, ...],
        events: AsyncGenerator[RAGStreamEvent, None],
        *,
        prepare_error: Exception | None = None,
    ) -> None:
        self.sources = sources
        self.events = events
        self.prepare_error = prepare_error

    async def stream_answer(
        self,
        question: str,
        *,
        session_id: UUID | None,
        document_ids: list[UUID] | None,
        top_k: int,
        top_n: int,
    ) -> RAGStream:
        del question, session_id, document_ids, top_k, top_n
        if self.prepare_error is not None:
            raise self.prepare_error
        return RAGStream(sources=self.sources, events=self.events)


async def stream_events(
    *events: RAGStreamEvent,
    error: Exception | None = None,
) -> AsyncGenerator[RAGStreamEvent, None]:
    for event in events:
        yield event
    if error is not None:
        raise error


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


async def test_chat_stream_endpoint_emits_sources_tokens_and_done(settings: Settings) -> None:
    session_id = uuid4()
    source = Citation(
        document_id=uuid4(),
        filename="policy.pdf",
        page=7,
        chunk_id=uuid4(),
        vector_score=0.93,
        rerank_score=None,
    )
    service = StubStreamingRAGService(
        (source,),
        stream_events(RAGStreamToken("Five "), RAGStreamToken("years"), RAGStreamDone(session_id)),
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
            response = await client.post("/api/v1/chat/stream", json={"question": "How long?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.text.count("event: token") == 2
    assert response.text.index("event: sources") < response.text.index("event: token")
    assert response.text.index("event: token") < response.text.index("event: done")
    assert str(source.document_id) in response.text
    assert str(session_id) in response.text


async def test_chat_stream_endpoint_serializes_runtime_error(settings: Settings) -> None:
    service = StubStreamingRAGService(
        (),
        stream_events(error=LLMServiceUnavailableError()),
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
            response = await client.post("/api/v1/chat/stream", json={"question": "Question"})

    assert response.status_code == 200
    assert "event: sources" in response.text
    assert "event: error" in response.text
    assert '"code":"llm_unavailable"' in response.text
    assert "event: done" not in response.text


async def test_chat_stream_preparation_error_keeps_http_status(settings: Settings) -> None:
    service = StubStreamingRAGService(
        (),
        stream_events(),
        prepare_error=ChatSessionNotFoundError(),
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
            response = await client.post("/api/v1/chat/stream", json={"question": "Question"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "chat_session_not_found"


async def test_chat_stream_disconnect_closes_upstream_generator(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    async def disconnectable_events() -> AsyncGenerator[RAGStreamEvent, None]:
        nonlocal closed
        try:
            yield RAGStreamToken("first")
            yield RAGStreamToken("second")
            yield RAGStreamDone(uuid4())
        finally:
            closed = True

    checks = iter([False, False, True])

    async def is_disconnected(request: Request) -> bool:
        del request
        return next(checks, True)

    monkeypatch.setattr(Request, "is_disconnected", is_disconnected)
    service = StubStreamingRAGService((), disconnectable_events())
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
            response = await client.post("/api/v1/chat/stream", json={"question": "Question"})

    assert "first" in response.text
    assert "second" not in response.text
    assert "event: done" not in response.text
    assert closed is True
