from collections.abc import AsyncIterator
from typing import cast

from fastapi import Request

from airgap_rag.api.dependencies.documents import SessionDatabase
from airgap_rag.db.repositories.indexing import SQLAlchemyIndexingRepository
from airgap_rag.embeddings.base import EmbeddingProvider
from airgap_rag.retrieval.service import RetrievalService
from airgap_rag.vector_store.base import VectorStore


async def get_retrieval_service(request: Request) -> AsyncIterator[RetrievalService]:
    database = cast(SessionDatabase, request.app.state.database)
    embedding_provider = cast(EmbeddingProvider, request.app.state.embedding_provider)
    vector_store = cast(VectorStore, request.app.state.vector_store)
    async with database.session() as session:
        yield RetrievalService(
            repository=SQLAlchemyIndexingRepository(session),
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
