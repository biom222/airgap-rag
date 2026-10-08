from sqlalchemy.ext.asyncio import AsyncSession

from airgap_rag.chunking.recursive import RecursiveCharacterChunker
from airgap_rag.core.config import Settings
from airgap_rag.db.repositories.indexing import SQLAlchemyIndexingRepository
from airgap_rag.documents.parsers import ParserRegistry
from airgap_rag.documents.storage import LocalDocumentStore
from airgap_rag.embeddings.base import EmbeddingProvider
from airgap_rag.indexing.service import IndexingService
from airgap_rag.vector_store.base import VectorStore


def create_indexing_service(
    settings: Settings,
    session: AsyncSession,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
) -> IndexingService:
    """Compose the indexing pipeline without coupling it to an HTTP endpoint."""
    return IndexingService(
        repository=SQLAlchemyIndexingRepository(session),
        store=LocalDocumentStore(
            root=settings.document_storage_path,
            max_upload_bytes=settings.document_max_upload_bytes,
            read_chunk_bytes=settings.document_read_chunk_bytes,
        ),
        parser_registry=ParserRegistry(),
        chunker=RecursiveCharacterChunker(settings.chunk_size, settings.chunk_overlap),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
