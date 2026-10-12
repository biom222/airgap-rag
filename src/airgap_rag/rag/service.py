# ruff: noqa: RUF001  # Russian fallback answer intentionally contains Cyrillic characters.

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID

from airgap_rag.core.tracing import tracer
from airgap_rag.db.repositories.chat import ChatRepository
from airgap_rag.llm.base import (
    LLMMessage,
    LLMModelNotFoundError,
    LLMProvider,
    LLMProviderError,
    LLMUnavailableError,
)
from airgap_rag.metrics import MetricsRecorder, NoOpMetrics
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
logger = logging.getLogger(__name__)
rag_tracer = tracer(__name__)


@dataclass(frozen=True, slots=True)
class _PreparedAnswer:
    question: str
    session_id: UUID | None
    messages: tuple[LLMMessage, ...] | None
    fallback_answer: str | None
    sources: tuple[Citation, ...]
    started_at: float


class RAGService:
    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        chat_repository: ChatRepository,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        history_limit: int,
        metrics: MetricsRecorder | None = None,
        trace_id: str | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._chat_repository = chat_repository
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider
        self._history_limit = history_limit
        self._metrics = metrics or NoOpMetrics()
        self._trace_id = trace_id

    async def answer(
        self,
        question: str,
        *,
        session_id: UUID | None,
        document_ids: list[UUID] | None,
        top_k: int,
        top_n: int,
    ) -> RAGAnswer:
        started_at = perf_counter()
        outcome = "error"
        try:
            prepared = await self._prepare(
                question,
                session_id=session_id,
                document_ids=document_ids,
                top_k=top_k,
                top_n=top_n,
                started_at=started_at,
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
            outcome = "success"
            return RAGAnswer(
                answer=answer,
                session_id=resolved_session_id,
                sources=prepared.sources,
            )
        finally:
            self._finish_rag("sync", outcome, started_at)

    async def stream_answer(
        self,
        question: str,
        *,
        session_id: UUID | None,
        document_ids: list[UUID] | None,
        top_k: int,
        top_n: int,
    ) -> RAGStream:
        started_at = perf_counter()
        try:
            prepared = await self._prepare(
                question,
                session_id=session_id,
                document_ids=document_ids,
                top_k=top_k,
                top_n=top_n,
                started_at=started_at,
            )
        except Exception:
            self._finish_rag("stream", "error", started_at)
            raise
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
        started_at: float,
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
            started_at=started_at,
        )

    async def _generate(self, messages: tuple[LLMMessage, ...]) -> str:
        started_at = perf_counter()
        try:
            with rag_tracer.start_as_current_span("rag.llm.generate") as span:
                span.set_attribute("llm.message_count", len(messages))
                answer = (await self._llm_provider.generate(messages)).strip()
        except (LLMUnavailableError, LLMModelNotFoundError) as exc:
            raise LLMServiceUnavailableError() from exc
        except LLMProviderError as exc:
            raise LLMGenerationFailedError() from exc
        finally:
            duration = perf_counter() - started_at
            self._metrics.observe_llm_generation(duration)
            logger.info(
                "llm_generation_completed",
                extra={
                    "trace_id": self._trace_id,
                    "mode": "sync",
                    "generation_duration_ms": round(duration * 1000, 3),
                },
            )
        if not answer:
            raise LLMGenerationFailedError()
        return answer

    async def _stream_prepared(
        self,
        prepared: _PreparedAnswer,
    ) -> AsyncGenerator[RAGStreamEvent, None]:
        outcome = "cancelled"
        try:
            if prepared.messages is None:
                answer = prepared.fallback_answer or INSUFFICIENT_CONTEXT_ANSWER
                yield RAGStreamToken(answer)
            else:
                parts: list[str] = []
                provider_stream = self._llm_provider.stream(prepared.messages)
                generation_started = perf_counter()
                first_token_recorded = False
                try:
                    async for text in provider_stream:
                        if text:
                            if not first_token_recorded:
                                ttft = perf_counter() - generation_started
                                self._metrics.observe_llm_ttft(ttft)
                                first_token_recorded = True
                            parts.append(text)
                            yield RAGStreamToken(text)
                except (LLMUnavailableError, LLMModelNotFoundError) as exc:
                    outcome = "error"
                    raise LLMServiceUnavailableError() from exc
                except LLMProviderError as exc:
                    outcome = "error"
                    raise LLMGenerationFailedError() from exc
                finally:
                    if isinstance(provider_stream, AsyncGenerator):
                        await provider_stream.aclose()
                    duration = perf_counter() - generation_started
                    self._metrics.observe_llm_generation(duration)
                    logger.info(
                        "llm_generation_completed",
                        extra={
                            "trace_id": self._trace_id,
                            "mode": "stream",
                            "generation_duration_ms": round(duration * 1000, 3),
                        },
                    )
                answer = "".join(parts).strip()
                if not answer:
                    outcome = "error"
                    raise LLMGenerationFailedError()

            resolved_session_id = await self._chat_repository.save_exchange(
                prepared.session_id,
                user_content=prepared.question,
                assistant_content=answer,
            )
            outcome = "success"
            yield RAGStreamDone(resolved_session_id)
        finally:
            self._finish_rag("stream", outcome, prepared.started_at)

    def _finish_rag(self, mode: str, outcome: str, started_at: float) -> None:
        duration = perf_counter() - started_at
        self._metrics.observe_rag(mode, outcome, duration)
        logger.info(
            "rag_request_completed",
            extra={
                "trace_id": self._trace_id,
                "mode": mode,
                "outcome": outcome,
                "duration_ms": round(duration * 1000, 3),
            },
        )
