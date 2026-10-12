import logging
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Protocol
from uuid import UUID

from airgap_rag.core.tracing import tracer
from airgap_rag.db.models.documents import DocumentChunk
from airgap_rag.embeddings.base import EmbeddingProvider
from airgap_rag.metrics import MetricsRecorder, NoOpMetrics
from airgap_rag.reranking.base import RerankDocument, Reranker, RerankerError
from airgap_rag.vector_store.base import VectorStore

logger = logging.getLogger(__name__)
retrieval_tracer = tracer(__name__)


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
        metrics: MetricsRecorder | None = None,
        trace_id: str | None = None,
        owner_id: str | None = None,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._reranker = reranker
        self._metrics = metrics or NoOpMetrics()
        self._trace_id = trace_id
        self._owner_id = owner_id

    async def search(
        self,
        question: str,
        *,
        top_k: int,
        top_n: int | None = None,
        document_ids: list[UUID] | None = None,
    ) -> list[RetrievedChunk]:
        embedding_started = perf_counter()
        with retrieval_tracer.start_as_current_span("rag.embedding") as span:
            span.set_attribute("rag.owner_scoped", self._owner_id is not None)
            vector = await self._embedding_provider.embed_query(question)
        embedding_duration = perf_counter() - embedding_started
        self._metrics.observe_embedding(embedding_duration)

        retrieval_started = perf_counter()
        with retrieval_tracer.start_as_current_span("rag.retrieval") as span:
            span.set_attribute("rag.top_k", top_k)
            await self._vector_store.ensure_collection(self._embedding_provider.dimension)
            hits = await self._vector_store.search(
                vector,
                limit=top_k,
                document_ids=document_ids,
                owner_id=self._owner_id,
            )
            chunks = await self._repository.get_chunks([hit.id for hit in hits])
            span.set_attribute("rag.hit_count", len(hits))

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
        retrieval_duration = perf_counter() - retrieval_started
        self._metrics.observe_retrieval(retrieval_duration)
        result_limit = min(top_n, len(results)) if top_n is not None else len(results)
        if self._reranker is None or not results:
            self._log_completed(
                embedding_duration=embedding_duration,
                retrieval_duration=retrieval_duration,
                reranking_duration=None,
                result_count=min(result_limit, len(results)),
            )
            return results[:result_limit]

        reranking_started = perf_counter()
        with retrieval_tracer.start_as_current_span("rag.reranking") as span:
            span.set_attribute("rag.candidate_count", len(results))
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
        reranking_duration = perf_counter() - reranking_started
        self._metrics.observe_reranking(reranking_duration)
        self._log_completed(
            embedding_duration=embedding_duration,
            retrieval_duration=retrieval_duration,
            reranking_duration=reranking_duration,
            result_count=min(result_limit, len(scored)),
        )
        return scored[:result_limit]

    def _log_completed(
        self,
        *,
        embedding_duration: float,
        retrieval_duration: float,
        reranking_duration: float | None,
        result_count: int,
    ) -> None:
        logger.info(
            "retrieval_completed",
            extra={
                "trace_id": self._trace_id,
                "embedding_duration_ms": round(embedding_duration * 1000, 3),
                "retrieval_duration_ms": round(retrieval_duration * 1000, 3),
                "reranking_duration_ms": (
                    round(reranking_duration * 1000, 3) if reranking_duration is not None else None
                ),
                "result_count": result_count,
            },
        )
