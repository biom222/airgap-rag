from collections.abc import AsyncIterator, Sequence
from uuid import UUID, uuid4

import pytest

from airgap_rag.llm.base import LLMMessage, LLMUnavailableError
from airgap_rag.rag.errors import ChatSessionNotFoundError, LLMServiceUnavailableError
from airgap_rag.rag.prompting import PromptBuilder
from airgap_rag.rag.service import INSUFFICIENT_CONTEXT_ANSWER, RAGService
from airgap_rag.rag.types import ChatHistoryMessage
from airgap_rag.retrieval.service import RetrievalService, RetrievedChunk


class StubRetrievalService(RetrievalService):
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    async def search(
        self,
        question: str,
        *,
        top_k: int,
        top_n: int | None = None,
        document_ids: list[UUID] | None = None,
    ) -> list[RetrievedChunk]:
        del question, top_k, document_ids
        return self.chunks[:top_n]


class FakeChatRepository:
    def __init__(self, history: list[ChatHistoryMessage] | None = None) -> None:
        self.history = history
        self.saved: list[tuple[UUID | None, str, str]] = []
        self.created_session_id = uuid4()

    async def get_history(
        self,
        session_id: UUID,
        *,
        limit: int,
    ) -> list[ChatHistoryMessage] | None:
        del session_id, limit
        return self.history

    async def save_exchange(
        self,
        session_id: UUID | None,
        *,
        user_content: str,
        assistant_content: str,
    ) -> UUID:
        self.saved.append((session_id, user_content, assistant_content))
        return session_id or self.created_session_id


class RecordingLLMProvider:
    def __init__(self, response: str = "Answer [S1]", error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.messages: Sequence[LLMMessage] = []

    async def generate(self, messages: Sequence[LLMMessage]) -> str:
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.response

    async def stream(self, messages: Sequence[LLMMessage]) -> AsyncIterator[str]:
        del messages
        yield self.response

    async def healthcheck(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def make_service(
    chunks: list[RetrievedChunk],
    repository: FakeChatRepository,
    llm: RecordingLLMProvider,
) -> RAGService:
    return RAGService(
        retrieval_service=StubRetrievalService(chunks),
        chat_repository=repository,
        prompt_builder=PromptBuilder(4000),
        llm_provider=llm,
        history_limit=10,
    )


async def test_rag_answer_returns_server_derived_citations() -> None:
    chunk = RetrievedChunk(
        document_id=uuid4(),
        filename="policy.pdf",
        page=4,
        chunk_id=uuid4(),
        chunk_index=1,
        text="Retention is five years.",
        vector_score=0.91,
        rerank_score=0.98,
    )
    repository = FakeChatRepository(history=[])
    llm = RecordingLLMProvider()
    service = make_service([chunk], repository, llm)

    result = await service.answer(
        "How long?",
        session_id=None,
        document_ids=[chunk.document_id],
        top_k=10,
        top_n=5,
    )

    assert result.answer == "Answer [S1]"
    assert result.sources[0].chunk_id == chunk.chunk_id
    assert result.sources[0].vector_score == 0.91
    assert result.sources[0].rerank_score == 0.98
    assert result.session_id == repository.created_session_id
    assert repository.saved == [(None, "How long?", "Answer [S1]")]
    assert "Retention is five years." in llm.messages[-1].content


async def test_rag_answer_does_not_call_llm_without_context() -> None:
    repository = FakeChatRepository(history=[])
    llm = RecordingLLMProvider(error=AssertionError("LLM must not be called"))
    service = make_service([], repository, llm)

    result = await service.answer(
        "Unknown?",
        session_id=None,
        document_ids=None,
        top_k=10,
        top_n=5,
    )

    assert result.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert result.sources == ()


async def test_rag_answer_rejects_unknown_session() -> None:
    service = make_service([], FakeChatRepository(history=None), RecordingLLMProvider())

    with pytest.raises(ChatSessionNotFoundError):
        await service.answer(
            "Question",
            session_id=uuid4(),
            document_ids=None,
            top_k=10,
            top_n=5,
        )


async def test_rag_answer_maps_llm_unavailable_error() -> None:
    chunk = RetrievedChunk(
        document_id=uuid4(),
        filename="policy.pdf",
        page=1,
        chunk_id=uuid4(),
        chunk_index=0,
        text="Context",
        vector_score=0.9,
    )
    service = make_service(
        [chunk],
        FakeChatRepository(history=[]),
        RecordingLLMProvider(error=LLMUnavailableError("offline")),
    )

    with pytest.raises(LLMServiceUnavailableError):
        await service.answer(
            "Question",
            session_id=None,
            document_ids=None,
            top_k=10,
            top_n=5,
        )
