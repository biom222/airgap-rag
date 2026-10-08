# ruff: noqa: RUF001  # Russian fallback answer intentionally contains Cyrillic characters.

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from uuid import UUID

from airgap_rag.db.repositories.chat import ChatRepository
from airgap_rag.llm.base import (
    LLMMessage,
    LLMModelNotFoundError,
    LLMProvider,
    LLMProviderError,
    LLMUnavailableError,
)
from airgap_rag.rag.errors import (
    ChatSessionNotFoundError,
    LLMGenerationFailedError,
    LLMServiceUnavailableError,
)
from airgap_rag.rag.prompting import PromptBuilder
from airgap_rag.rag.types import (
    ChatHistoryMessage,
    Citation,
    RAGAnswer,
    RAGStream,
    RAGStreamDone,
    RAGStreamEvent,
    RAGStreamToken,
)
from airgap_rag.retrieval.service import RetrievalService

INSUFFICIENT_CONTEXT_ANSWER = "В предоставленных документах недостаточно информации для ответа."


@dataclass(frozen=True, slots=True)
class _PreparedAnswer:
    question: str
    session_id: UUID | None
    messages: tuple[LLMMessage, ...] | None
    fallback_answer: str | None
    sources: tuple[Citation, ...]


class RAGService:
    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        chat_repository: ChatRepository,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        history_limit: int,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._chat_repository = chat_repository
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider
        self._history_limit = history_limit

    async def answer(
        self,
        question: str,
        *,
        session_id: UUID | None,
        document_ids: list[UUID] | None,
        top_k: int,
        top_n: int,
    ) -> RAGAnswer:
        prepared = await self._prepare(
            question,
            session_id=session_id,
            document_ids=document_ids,
            top_k=top_k,
            top_n=top_n,
        )
        if prepared.messages is None:
            answer = prepared.fallback_answer or INSUFFICIENT_CONTEXT_ANSWER
        else:
            answer = await self._generate(prepared.messages)

        resolved_session_id = await self._chat_repository.save_exchange(
            session_id,
            user_content=question,
            assistant_content=answer,
        )
        return RAGAnswer(
            answer=answer,
            session_id=resolved_session_id,
            sources=prepared.sources,
        )

    async def stream_answer(
        self,
        question: str,
        *,
        session_id: UUID | None,
        document_ids: list[UUID] | None,
        top_k: int,
        top_n: int,
    ) -> RAGStream:
        prepared = await self._prepare(
            question,
            session_id=session_id,
            document_ids=document_ids,
            top_k=top_k,
            top_n=top_n,
        )
        return RAGStream(
            sources=prepared.sources,
            events=self._stream_prepared(prepared),
        )

    async def _prepare(
        self,
        question: str,
        *,
        session_id: UUID | None,
        document_ids: list[UUID] | None,
        top_k: int,
        top_n: int,
    ) -> _PreparedAnswer:
        history: list[ChatHistoryMessage] = []
        if session_id is not None:
            loaded_history = await self._chat_repository.get_history(
                session_id,
                limit=self._history_limit,
            )
            if loaded_history is None:
                raise ChatSessionNotFoundError()
            history = loaded_history

        chunks = await self._retrieval_service.search(
            question,
            top_k=top_k,
            top_n=top_n,
            document_ids=document_ids,
        )
        built_prompt = self._prompt_builder.build(question, chunks, history) if chunks else None
        source_chunks = built_prompt.context_chunks if built_prompt is not None else ()
        sources = tuple(
            Citation(
                document_id=chunk.document_id,
                filename=chunk.filename,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                vector_score=chunk.vector_score,
                rerank_score=chunk.rerank_score,
            )
            for chunk in source_chunks
        )
        return _PreparedAnswer(
            question=question,
            session_id=session_id,
            messages=built_prompt.messages if built_prompt is not None else None,
            fallback_answer=None if built_prompt is not None else INSUFFICIENT_CONTEXT_ANSWER,
            sources=sources,
        )

    async def _generate(self, messages: tuple[LLMMessage, ...]) -> str:
        try:
            answer = (await self._llm_provider.generate(messages)).strip()
        except (LLMUnavailableError, LLMModelNotFoundError) as exc:
            raise LLMServiceUnavailableError() from exc
        except LLMProviderError as exc:
            raise LLMGenerationFailedError() from exc
        if not answer:
            raise LLMGenerationFailedError()
        return answer

    async def _stream_prepared(
        self,
        prepared: _PreparedAnswer,
    ) -> AsyncGenerator[RAGStreamEvent, None]:
        if prepared.messages is None:
            answer = prepared.fallback_answer or INSUFFICIENT_CONTEXT_ANSWER
            yield RAGStreamToken(answer)
        else:
            parts: list[str] = []
            provider_stream = self._llm_provider.stream(prepared.messages)
            try:
                async for text in provider_stream:
                    if text:
                        parts.append(text)
                        yield RAGStreamToken(text)
            except (LLMUnavailableError, LLMModelNotFoundError) as exc:
                raise LLMServiceUnavailableError() from exc
            except LLMProviderError as exc:
                raise LLMGenerationFailedError() from exc
            finally:
                if isinstance(provider_stream, AsyncGenerator):
                    await provider_stream.aclose()
            answer = "".join(parts).strip()
            if not answer:
                raise LLMGenerationFailedError()

        resolved_session_id = await self._chat_repository.save_exchange(
            prepared.session_id,
            user_content=prepared.question,
            assistant_content=answer,
        )
        yield RAGStreamDone(resolved_session_id)
