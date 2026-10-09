import hashlib
import time
from collections.abc import Sequence
from dataclasses import asdict
from uuid import UUID

from airgap_rag.benchmarks.llm import BenchmarkError
from airgap_rag.benchmarks.models import RAGReport, RAGSample, summarize
from airgap_rag.core.config import Settings
from airgap_rag.db.repositories.indexing import SessionChunkRepository
from airgap_rag.db.session import Database
from airgap_rag.embeddings.base import EmbeddingProvider
from airgap_rag.embeddings.factory import create_embedding_provider
from airgap_rag.llm.base import LLMProvider
from airgap_rag.llm.factory import create_llm_provider
from airgap_rag.rag.prompting import PromptBuilder
from airgap_rag.reranking.base import RerankDocument, Reranker, RerankResult
from airgap_rag.reranking.factory import create_reranker
from airgap_rag.retrieval.service import RetrievalService
from airgap_rag.system_info import collect_hardware_info
from airgap_rag.vector_store.base import VectorStore
from airgap_rag.vector_store.factory import create_vector_store


class TimedEmbeddingProvider:
    def __init__(self, inner: EmbeddingProvider) -> None:
        self._inner = inner
        self.query_latency_ms = 0.0

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._inner.embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        started = time.perf_counter_ns()
        try:
            return await self._inner.embed_query(text)
        finally:
            self.query_latency_ms = (time.perf_counter_ns() - started) / 1_000_000

    async def healthcheck(self) -> bool:
        return await self._inner.healthcheck()


class TimedReranker:
    def __init__(self, inner: Reranker) -> None:
        self._inner = inner
        self.latency_ms = 0.0

    async def rerank(
        self,
        query: str,
        documents: Sequence[RerankDocument],
    ) -> list[RerankResult]:
        started = time.perf_counter_ns()
        try:
            return await self._inner.rerank(query, documents)
        finally:
            self.latency_ms = (time.perf_counter_ns() - started) / 1_000_000

    async def healthcheck(self) -> bool:
        return await self._inner.healthcheck()


class RAGBenchmark:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        reranker: Reranker | None,
    ) -> None:
        self._settings = settings
        self._database = database
        self._embedding = TimedEmbeddingProvider(embedding_provider)
        self._reranker = TimedReranker(reranker) if reranker is not None else None
        self._vector_store = vector_store
        self._llm = llm_provider
        self._retrieval = RetrievalService(
            SessionChunkRepository(database),
            self._embedding,
            vector_store,
            self._reranker,
        )
        self._prompt_builder = PromptBuilder(settings.rag_context_max_characters)

    async def close(self) -> None:
        await self._llm.close()
        await self._vector_store.close()
        await self._database.dispose()

    async def measure(self, question: str, document_ids: list[UUID] | None) -> RAGSample:
        total_started = time.perf_counter_ns()
        retrieval_started = time.perf_counter_ns()
        chunks = await self._retrieval.search(
            question,
            top_k=self._settings.retrieval_top_k,
            top_n=(self._settings.reranker_top_n if self._settings.reranker_enabled else None),
            document_ids=document_ids,
        )
        retrieval_total_ms = (time.perf_counter_ns() - retrieval_started) / 1_000_000
        if not chunks:
            raise BenchmarkError("RAG benchmark retrieved no chunks")
        built_prompt = self._prompt_builder.build(question, chunks, [])
        generation_started = time.perf_counter_ns()
        await self._llm.generate(built_prompt.messages)
        generation_ms = (time.perf_counter_ns() - generation_started) / 1_000_000
        reranking_ms = self._reranker.latency_ms if self._reranker else 0.0
        retrieval_only_ms = max(
            0.0,
            retrieval_total_ms - self._embedding.query_latency_ms - reranking_ms,
        )
        return RAGSample(
            embedding_latency_ms=self._embedding.query_latency_ms,
            retrieval_latency_ms=retrieval_only_ms,
            reranking_latency_ms=reranking_ms,
            generation_latency_ms=generation_ms,
            total_latency_ms=(time.perf_counter_ns() - total_started) / 1_000_000,
            retrieved_chunks=len(chunks),
        )

    async def run(
        self,
        *,
        question: str,
        document_ids: list[UUID] | None,
        warmup_iterations: int,
        measured_iterations: int,
    ) -> RAGReport:
        if warmup_iterations < 0 or measured_iterations < 1:
            raise ValueError("warmup must be non-negative and iterations must be positive")
        for _ in range(warmup_iterations):
            await self.measure(question, document_ids)
        samples = [await self.measure(question, document_ids) for _ in range(measured_iterations)]
        settings = self._settings
        embedding_model = (
            settings.embedding_model_name_or_path
            if settings.embedding_provider == "sentence_transformer"
            else f"mock:{settings.embedding_dimension}"
        )
        reranker = None
        if settings.reranker_enabled:
            reranker = (
                settings.reranker_model_name_or_path
                if settings.reranker_provider == "cross_encoder"
                else "mock"
            )
        return RAGReport(
            hardware=asdict(collect_hardware_info()),
            llm_provider=settings.llm_provider,
            llm_model=settings.ollama_model if settings.llm_provider == "ollama" else "mock",
            embedding_provider=settings.embedding_provider,
            embedding_model=embedding_model,
            reranker=reranker,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            question_sha256=hashlib.sha256(question.encode()).hexdigest(),
            samples=samples,
            embedding_latency_ms=summarize([item.embedding_latency_ms for item in samples]),
            retrieval_latency_ms=summarize([item.retrieval_latency_ms for item in samples]),
            reranking_latency_ms=summarize([item.reranking_latency_ms for item in samples]),
            generation_latency_ms=summarize([item.generation_latency_ms for item in samples]),
            total_latency_ms=summarize([item.total_latency_ms for item in samples]),
        )


def create_rag_benchmark(settings: Settings) -> RAGBenchmark:
    database = Database(settings)
    return RAGBenchmark(
        settings=settings,
        database=database,
        embedding_provider=create_embedding_provider(settings),
        vector_store=create_vector_store(settings),
        llm_provider=create_llm_provider(settings),
        reranker=create_reranker(settings),
    )
