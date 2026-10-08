from airgap_rag.core.config import Settings
from airgap_rag.embeddings.base import EmbeddingProvider
from airgap_rag.embeddings.mock import MockEmbeddingProvider
from airgap_rag.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "mock":
        return MockEmbeddingProvider(settings.embedding_dimension)
    return SentenceTransformerEmbeddingProvider(
        settings.embedding_model_name_or_path,
        airgap_mode=settings.airgap_mode,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        document_prefix=settings.embedding_document_prefix,
        query_prefix=settings.embedding_query_prefix,
    )
