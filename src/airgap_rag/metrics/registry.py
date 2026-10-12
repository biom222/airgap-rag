from typing import Protocol

from prometheus_client import CollectorRegistry, Counter, Histogram


class MetricsRecorder(Protocol):
    def observe_http(self, method: str, route: str, status_code: int, duration: float) -> None: ...

    def record_document_created(self) -> None: ...

    def observe_embedding(self, duration: float) -> None: ...

    def observe_retrieval(self, duration: float) -> None: ...

    def observe_reranking(self, duration: float) -> None: ...

    def observe_rag(self, mode: str, outcome: str, duration: float) -> None: ...

    def observe_llm_ttft(self, duration: float) -> None: ...

    def observe_llm_generation(self, duration: float) -> None: ...

    def add_llm_tokens(self, count: int) -> None: ...

    def record_ingestion_failure(self) -> None: ...

    def record_task_failure(self, retryable: bool) -> None: ...


class NoOpMetrics:
    def observe_http(self, method: str, route: str, status_code: int, duration: float) -> None:
        return None

    def record_document_created(self) -> None:
        return None

    def observe_embedding(self, duration: float) -> None:
        return None

    def observe_retrieval(self, duration: float) -> None:
        return None

    def observe_reranking(self, duration: float) -> None:
        return None

    def observe_rag(self, mode: str, outcome: str, duration: float) -> None:
        return None

    def observe_llm_ttft(self, duration: float) -> None:
        return None

    def observe_llm_generation(self, duration: float) -> None:
        return None

    def add_llm_tokens(self, count: int) -> None:
        return None

    def record_ingestion_failure(self) -> None:
        return None

    def record_task_failure(self, retryable: bool) -> None:
        return None


class ApplicationMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.http_requests = Counter(
            "http_requests_total",
            "HTTP requests completed by the API.",
            ("method", "route", "status_code"),
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds.",
            ("method", "route"),
            registry=self.registry,
        )
        self.documents = Counter(
            "documents_total",
            "Documents created after deduplication.",
            registry=self.registry,
        )
        self.document_ingestion_failures = Counter(
            "document_ingestion_failures_total",
            "Document ingestion jobs that reached a terminal failure.",
            registry=self.registry,
        )
        self.rag_requests = Counter(
            "rag_requests_total",
            "RAG requests completed.",
            ("mode", "outcome"),
            registry=self.registry,
        )
        self.rag_duration = Histogram(
            "rag_request_duration_seconds",
            "End-to-end RAG request duration in seconds.",
            ("mode",),
            registry=self.registry,
        )
        self.embedding_duration = Histogram(
            "embedding_duration_seconds",
            "Query embedding duration in seconds.",
            registry=self.registry,
        )
        self.retrieval_duration = Histogram(
            "retrieval_duration_seconds",
            "Vector search and chunk loading duration in seconds.",
            registry=self.registry,
        )
        self.reranking_duration = Histogram(
            "reranking_duration_seconds",
            "Reranking duration in seconds.",
            registry=self.registry,
        )
        self.llm_ttft = Histogram(
            "llm_ttft_seconds",
            "Time to first streamed LLM token in seconds.",
            registry=self.registry,
        )
        self.llm_generation_duration = Histogram(
            "llm_generation_duration_seconds",
            "LLM generation duration in seconds.",
            registry=self.registry,
        )
        self.llm_tokens_generated = Counter(
            "llm_tokens_generated_total",
            "Output tokens reported by the LLM runtime.",
            registry=self.registry,
        )
        self.task_failures = Counter(
            "task_failures_total",
            "Background task execution failures.",
            ("retryable",),
            registry=self.registry,
        )

    def observe_http(self, method: str, route: str, status_code: int, duration: float) -> None:
        status = str(status_code)
        self.http_requests.labels(method=method, route=route, status_code=status).inc()
        self.http_duration.labels(method=method, route=route).observe(duration)

    def record_document_created(self) -> None:
        self.documents.inc()

    def observe_embedding(self, duration: float) -> None:
        self.embedding_duration.observe(duration)

    def observe_retrieval(self, duration: float) -> None:
        self.retrieval_duration.observe(duration)

    def observe_reranking(self, duration: float) -> None:
        self.reranking_duration.observe(duration)

    def observe_rag(self, mode: str, outcome: str, duration: float) -> None:
        self.rag_requests.labels(mode=mode, outcome=outcome).inc()
        self.rag_duration.labels(mode=mode).observe(duration)

    def observe_llm_ttft(self, duration: float) -> None:
        self.llm_ttft.observe(duration)

    def observe_llm_generation(self, duration: float) -> None:
        self.llm_generation_duration.observe(duration)

    def add_llm_tokens(self, count: int) -> None:
        if count > 0:
            self.llm_tokens_generated.inc(count)

    def record_ingestion_failure(self) -> None:
        self.document_ingestion_failures.inc()

    def record_task_failure(self, retryable: bool) -> None:
        self.task_failures.labels(retryable=str(retryable).lower()).inc()
