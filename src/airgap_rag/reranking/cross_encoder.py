from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from anyio import to_thread

from airgap_rag.reranking.base import RerankDocument, RerankerError, RerankResult


class LocalCrossEncoderReranker:
    """Adapter for a local sentence-transformers CrossEncoder model."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        airgap_mode: bool,
        device: str,
        batch_size: int,
    ) -> None:
        if airgap_mode and not Path(model_name_or_path).is_dir():
            raise RerankerError(
                "AIRGAP_MODE requires RERANKER_MODEL_NAME_OR_PATH to be a local directory"
            )
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "Install the reranking dependency: pip install -e '.[reranking]'"
            ) from exc

        try:
            self._model: Any = CrossEncoder(
                model_name_or_path,
                device=device,
                local_files_only=airgap_mode,
            )
        except Exception as exc:
            raise RerankerError("Failed to load the local reranker model") from exc
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        documents: Sequence[RerankDocument],
    ) -> list[RerankResult]:
        if not documents:
            return []

        def predict() -> list[float]:
            scores = self._model.predict(
                [(query, document.text) for document in documents],
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [float(score) for score in scores]

        try:
            scores = await to_thread.run_sync(predict)
        except Exception as exc:
            raise RerankerError("Local reranker inference failed") from exc
        if len(scores) != len(documents):
            raise RerankerError("Local reranker returned an unexpected score count")
        return [
            RerankResult(id=document.id, score=score)
            for document, score in zip(documents, scores, strict=True)
        ]

    async def healthcheck(self) -> bool:
        try:
            results = await self.rerank(
                "healthcheck",
                [RerankDocument(id=UUID(int=0), text="healthcheck")],
            )
        except Exception:  # Provider boundary: readiness must return a boolean.
            return False
        return len(results) == 1
