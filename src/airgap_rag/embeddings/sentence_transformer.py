from pathlib import Path
from typing import Any

from anyio import to_thread

from airgap_rag.embeddings.base import EmbeddingProviderError


class SentenceTransformerEmbeddingProvider:
    """Adapter for a local sentence-transformers model."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        airgap_mode: bool,
        device: str,
        batch_size: int,
        document_prefix: str,
        query_prefix: str,
    ) -> None:
        if airgap_mode and not Path(model_name_or_path).is_dir():
            raise EmbeddingProviderError(
                "AIRGAP_MODE requires EMBEDDING_MODEL_NAME_OR_PATH to be a local directory"
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingProviderError(
                "Install the optional embeddings dependency: pip install -e '.[embeddings]'"
            ) from exc

        try:
            self._model: Any = SentenceTransformer(
                model_name_or_path,
                device=device,
                local_files_only=airgap_mode,
            )
        except Exception as exc:
            raise EmbeddingProviderError("Failed to load the local embedding model") from exc

        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None or dimension <= 0:
            raise EmbeddingProviderError("Embedding model did not report a valid dimension")
        self._dimension = int(dimension)
        self._batch_size = batch_size
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prepared = [f"{self._document_prefix}{text}" for text in texts]
        return await self._encode(prepared)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._encode([f"{self._query_prefix}{text}"])
        return vectors[0]

    async def healthcheck(self) -> bool:
        try:
            vector = await self.embed_query("healthcheck")
        except Exception:  # Provider boundary: readiness must return a boolean.
            return False
        return len(vector) == self._dimension

    async def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        def encode() -> list[list[float]]:
            result = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [[float(value) for value in row] for row in result]

        vectors = await to_thread.run_sync(encode)
        if any(len(vector) != self._dimension for vector in vectors):
            raise EmbeddingProviderError("Embedding model returned an unexpected dimension")
        return vectors
