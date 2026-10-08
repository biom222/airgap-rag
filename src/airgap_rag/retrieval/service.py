import logging
from dataclasses import dataclass, replace
from typing import Protocol
from uuid import UUID

from airgap_rag.db.models.documents import DocumentChunk
from airgap_rag.embeddings.base import EmbeddingProvider
from airgap_rag.reranking.base import RerankDocument, Reranker, RerankerError
from airgap_rag.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    document_id: UUID
    filename: str
    page: int | None
    chunk_id: UUID
    chunk_index: int
    text: str
    vector_score: float
    rerank_score: float | None = None


class ChunkRepository(Protocol):
    async def get_chunks(self, chunk_ids: list[UUID]) -> dict[UUID, DocumentChunk]: ...


class RetrievalService:
    def __init__(
        self,
        repository: ChunkRepository,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._reranker = reranker

    async def search(
        self,
        question: str,
        *,
        top_k: int,
        top_n: int | None = None,
        document_ids: list[UUID] | None = None,
    ) -> list[RetrievedChunk]:
        vector = await self._embedding_provider.embed_query(question)
        await self._vector_store.ensure_collection(self._embedding_provider.dimension)
        hits = await self._vector_store.search(
            vector,
            limit=top_k,
            document_ids=document_ids,
        )
        chunks = await self._repository.get_chunks([hit.id for hit in hits])

        results: list[RetrievedChunk] = []
        for hit in hits:
            chunk = chunks.get(hit.id)
            if chunk is None:
                logger.warning("retrieved_chunk_missing", extra={"chunk_id": str(hit.id)})
                continue
            filename = hit.payload.get("filename")
            if not isinstance(filename, str) or not filename:
                logger.warning("retrieved_payload_invalid", extra={"chunk_id": str(hit.id)})
                continue
            results.append(
                RetrievedChunk(
                    document_id=chunk.document_id,
                    filename=filename,
                    page=chunk.page,
                    chunk_id=chunk.id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    vector_score=hit.score,
                )
            )
        result_limit = min(top_n, len(results)) if top_n is not None else len(results)
        if self._reranker is None or not results:
            return results[:result_limit]

        reranked = await self._reranker.rerank(
            question,
            [RerankDocument(id=result.chunk_id, text=result.text) for result in results],
        )
        scores = {result.id: result.score for result in reranked}
        expected_ids = {result.chunk_id for result in results}
        if set(scores) != expected_ids:
            raise RerankerError("Reranker results do not match retrieved chunks")
        scored = [replace(result, rerank_score=scores[result.chunk_id]) for result in results]
        scored.sort(
            key=lambda result: (
                result.rerank_score if result.rerank_score is not None else float("-inf")
            ),
            reverse=True,
        )
        return scored[:result_limit]
