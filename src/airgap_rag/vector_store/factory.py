from qdrant_client import AsyncQdrantClient

from airgap_rag.core.config import Settings
from airgap_rag.vector_store.qdrant import QdrantVectorStore


def create_vector_store(settings: Settings) -> QdrantVectorStore:
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=settings.qdrant_timeout_seconds,
    )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
