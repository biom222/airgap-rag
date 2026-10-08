import math
from pathlib import Path

import pytest

from airgap_rag.embeddings.base import EmbeddingProviderError
from airgap_rag.embeddings.mock import MockEmbeddingProvider
from airgap_rag.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider


async def test_mock_embeddings_are_deterministic_and_normalized() -> None:
    provider = MockEmbeddingProvider(dimension=12)

    first = await provider.embed_query("same text")
    second = await provider.embed_query("same text")

    assert first == second
    assert len(first) == 12
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)


async def test_mock_documents_preserve_input_order() -> None:
    provider = MockEmbeddingProvider(dimension=8)

    vectors = await provider.embed_documents(["first", "second"])

    assert len(vectors) == 2
    assert vectors[0] != vectors[1]


def test_sentence_transformer_requires_local_model_in_airgap(runtime_path: Path) -> None:
    missing = runtime_path / "missing-model"

    with pytest.raises(EmbeddingProviderError, match="local directory"):
        SentenceTransformerEmbeddingProvider(
            str(missing),
            airgap_mode=True,
            device="cpu",
            batch_size=8,
            document_prefix="passage: ",
            query_prefix="query: ",
        )
