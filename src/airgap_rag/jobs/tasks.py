import asyncio
import logging
from dataclasses import dataclass
from uuid import UUID

from opentelemetry import context as otel_context
from opentelemetry import propagate
from prometheus_client import start_http_server
from taskiq import TaskiqEvents, TaskiqState

from airgap_rag.core.config import Settings, get_settings
from airgap_rag.core.logging import configure_logging
from airgap_rag.core.tracing import TracingRuntime, configure_tracing, tracer
from airgap_rag.db.repositories.jobs import SQLAlchemyJobRepository
from airgap_rag.db.session import Database
from airgap_rag.embeddings.base import EmbeddingProvider
from airgap_rag.embeddings.factory import create_embedding_provider
from airgap_rag.indexing.factory import create_indexing_service
from airgap_rag.jobs.broker import broker
from airgap_rag.jobs.errors import RetryableJobError
from airgap_rag.jobs.service import JobExecutionService, JobPublisher
from airgap_rag.metrics import ApplicationMetrics
from airgap_rag.vector_store.base import VectorStore
from airgap_rag.vector_store.factory import create_vector_store


@dataclass(slots=True)
class WorkerRuntime:
    settings: Settings
    database: Database
    embedding_provider: EmbeddingProvider
    vector_store: VectorStore
    tracing_runtime: TracingRuntime | None = None

    async def close(self) -> None:
        await self.vector_store.close()
        await self.database.dispose()
        if self.tracing_runtime is not None:
            self.tracing_runtime.shutdown()


_runtime: WorkerRuntime | None = None
_runtime_lock = asyncio.Lock()
worker_metrics = ApplicationMetrics()
logger = logging.getLogger(__name__)


async def get_worker_runtime() -> WorkerRuntime:
    global _runtime
    if _runtime is not None:
        return _runtime
    async with _runtime_lock:
        if _runtime is None:
            settings = get_settings()
            _runtime = WorkerRuntime(
                settings=settings,
                database=Database(settings),
                embedding_provider=create_embedding_provider(settings),
                vector_store=create_vector_store(settings),
            )
            _runtime.tracing_runtime = configure_tracing(
                settings,
                sqlalchemy_engine=_runtime.database.engine.sync_engine,
                service_name="airgap-rag-worker",
            )
    return _runtime


@broker.task(
    task_name="airgap_rag.jobs.ingest_document",
    retry_on_error=True,
    max_retries=get_settings().job_max_attempts - 1,
    delay=get_settings().job_retry_delay_seconds,
    types_of_exceptions=(RetryableJobError,),
)
async def ingest_document_task(
    job_id: str,
    trace_context: dict[str, str] | None = None,
) -> str:
    runtime = await get_worker_runtime()
    parsed_job_id = UUID(job_id)
    parent_context = propagate.extract(trace_context or {})
    token = otel_context.attach(parent_context)
    try:
        with tracer(__name__).start_as_current_span("ingestion_job") as span:
            span.set_attribute("job.id", job_id)
            async with runtime.database.session() as session:
                service = JobExecutionService(
                    repository=SQLAlchemyJobRepository(session),
                    indexing_service=create_indexing_service(
                        runtime.settings,
                        session,
                        runtime.embedding_provider,
                        runtime.vector_store,
                        job_id=parsed_job_id,
                    ),
                    max_attempts=runtime.settings.job_max_attempts,
                    timeout_seconds=runtime.settings.job_timeout_seconds,
                    stale_after_seconds=runtime.settings.job_stale_after_seconds,
                    metrics=worker_metrics,
                )
                return (await service.execute(parsed_job_id)).value
    finally:
        otel_context.detach(token)


class TaskiqJobPublisher(JobPublisher):
    async def enqueue_ingestion(
        self,
        job_id: UUID,
        *,
        trace_context: dict[str, str] | None = None,
    ) -> None:
        await ingest_document_task.kiq(str(job_id), trace_context)


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def start_worker_observability(state: TaskiqState) -> None:
    del state
    settings = get_settings()
    configure_logging(settings.log_level)
    start_http_server(
        port=settings.worker_metrics_port,
        addr=settings.worker_metrics_host,
        registry=worker_metrics.registry,
    )
    logger.info(
        "worker_metrics_started",
        extra={
            "host": settings.worker_metrics_host,
            "port": settings.worker_metrics_port,
        },
    )


@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def close_worker_runtime(state: TaskiqState) -> None:
    del state
    global _runtime
    if _runtime is not None:
        await _runtime.close()
        _runtime = None
