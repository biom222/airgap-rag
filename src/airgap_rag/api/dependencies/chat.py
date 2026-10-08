from typing import cast

from fastapi import Request

from airgap_rag.core.config import Settings
from airgap_rag.db.protocols import AsyncSessionProvider
from airgap_rag.db.repositories.chat import SessionChatRepository
from airgap_rag.db.repositories.indexing import SessionChunkRepository
from airgap_rag.embeddings.base import EmbeddingProvider
from airgap_rag.llm.base import LLMProvider
from airgap_rag.rag.prompting import PromptBuilder
from airgap_rag.rag.service import RAGService
from airgap_rag.reranking.base import Reranker
from airgap_rag.retrieval.service import RetrievalService
from airgap_rag.vector_store.base import VectorStore


def get_chat_service(request: Request) -> RAGService:
    settings = cast(Settings, request.app.state.settings)
    database = cast(AsyncSessionProvider, request.app.state.database)
    embedding_provider = cast(EmbeddingProvider, request.app.state.embedding_provider)
    vector_store = cast(VectorStore, request.app.state.vector_store)
    reranker = cast(Reranker | None, request.app.state.reranker)
    llm_provider = cast(LLMProvider, request.app.state.llm_provider)
    retrieval_service = RetrievalService(
        repository=SessionChunkRepository(database),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker=reranker,
    )
    return RAGService(
        retrieval_service=retrieval_service,
        chat_repository=SessionChatRepository(database),
        prompt_builder=PromptBuilder(settings.rag_context_max_characters),
        llm_provider=llm_provider,
        history_limit=settings.chat_history_max_messages,
    )
