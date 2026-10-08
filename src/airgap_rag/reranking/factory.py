from airgap_rag.core.config import Settings
from airgap_rag.reranking.base import Reranker
from airgap_rag.reranking.cross_encoder import LocalCrossEncoderReranker
from airgap_rag.reranking.mock import MockReranker


def create_reranker(settings: Settings) -> Reranker | None:
    if not settings.reranker_enabled:
        return None
    if settings.reranker_provider == "mock":
        return MockReranker()
    return LocalCrossEncoderReranker(
        settings.reranker_model_name_or_path,
        airgap_mode=settings.airgap_mode,
        device=settings.reranker_device,
        batch_size=settings.reranker_batch_size,
    )
