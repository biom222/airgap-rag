import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Protocol

from fastapi import FastAPI

from airgap_rag.core.config import Settings
from airgap_rag.embeddings.base import EmbeddingProvider
from airgap_rag.embeddings.factory import create_embedding_provider
from airgap_rag.vector_store.base import VectorStore
from airgap_rag.vector_store.factory import create_vector_store

logger = logging.getLogger(__name__)


class DisposableDatabase(Protocol):
    async def dispose(self) -> None: ...


def create_lifespan(
    settings: Settings,
    database: DisposableDatabase,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    resolved_embedding_provider = embedding_provider or create_embedding_provider(settings)
    resolved_vector_store = vector_store or create_vector_store(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.database = database
        app.state.embedding_provider = resolved_embedding_provider
        app.state.vector_store = resolved_vector_store
        logger.info("application_started", extra={"app_env": settings.app_env})
        try:
            yield
        finally:
            await resolved_vector_store.close()
            await database.dispose()
            logger.info("application_stopped")

    return lifespan
