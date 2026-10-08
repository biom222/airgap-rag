# ruff: noqa: RUF001  # Russian fallback answer intentionally contains Cyrillic characters.

from uuid import UUID

from airgap_rag.db.repositories.chat import ChatRepository
from airgap_rag.llm.base import (
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
from airgap_rag.rag.types import Citation, RAGAnswer
from airgap_rag.retrieval.service import RetrievalService

INSUFFICIENT_CONTEXT_ANSWER = "В предоставленных документах недостаточно информации для ответа."


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
        history = []
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
        if chunks:
            built_prompt = self._prompt_builder.build(question, chunks, history)
            try:
                answer = (await self._llm_provider.generate(built_prompt.messages)).strip()
            except (LLMUnavailableError, LLMModelNotFoundError) as exc:
                raise LLMServiceUnavailableError() from exc
            except LLMProviderError as exc:
                raise LLMGenerationFailedError() from exc
            if not answer:
                raise LLMGenerationFailedError()
            source_chunks = built_prompt.context_chunks
        else:
            answer = INSUFFICIENT_CONTEXT_ANSWER
            source_chunks = ()

        resolved_session_id = await self._chat_repository.save_exchange(
            session_id,
            user_content=question,
            assistant_content=answer,
        )
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
        return RAGAnswer(answer=answer, session_id=resolved_session_id, sources=sources)
