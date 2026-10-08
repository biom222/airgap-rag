from uuid import uuid4

import pytest

from airgap_rag.core.config import Settings
from airgap_rag.reranking.base import RerankDocument, RerankerError
from airgap_rag.reranking.factory import create_reranker
from airgap_rag.reranking.mock import MockReranker


def test_reranker_is_disabled_by_default() -> None:
    assert create_reranker(Settings(reranker_enabled=False)) is None


async def test_mock_reranker_scores_lexical_overlap() -> None:
    reranker = MockReranker()
    matching_id = uuid4()
    results = await reranker.rerank(
        "retention period",
        [
            RerankDocument(id=matching_id, text="The retention period is five years."),
            RerankDocument(id=uuid4(), text="Unrelated content."),
        ],
    )

    assert results[0].id == matching_id
    assert results[0].score == 1.0
    assert results[1].score == 0.0


def test_cross_encoder_requires_local_directory_in_airgap_mode() -> None:
    settings = Settings(
        reranker_enabled=True,
        reranker_provider="cross_encoder",
        reranker_model_name_or_path=".tmp/nonexistent-reranker-model",
        airgap_mode=True,
    )

    with pytest.raises(RerankerError, match="local directory"):
        create_reranker(settings)
