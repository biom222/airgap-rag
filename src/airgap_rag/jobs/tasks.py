import asyncio
from dataclasses import dataclass
from uuid import UUID

from taskiq import TaskiqEvents, TaskiqState

from airgap_rag.core.config import Settings, get_settings
from airgap_rag.db.repositories.jobs import SQLAlchemyJobRepository
from airgap_rag.db.session import Database
from airgap_rag.embeddings.base import EmbeddingProvider
from airgap_rag.embeddings.factory import create_embedding_provider
from airgap_rag.indexing.factory import create_indexing_service
from airgap_rag.jobs.broker import broker
from airgap_rag.jobs.errors import RetryableJobError
from airgap_rag.jobs.service import JobExecutionService, JobPublisher
from airgap_rag.vector_store.base import VectorStore
from airgap_rag.vector_store.factory import create_vector_store


@dataclass(slots=True)
class WorkerRuntime:
    settings: Settings
    database: Database
    embedding_provider: EmbeddingProvider
    vector_store: VectorStore

    async def close(self) -> None:
        await self.vector_store.close()
        await self.database.dispose()


_runtime: WorkerRuntime | None = None
_runtime_lock = asyncio.Lock()


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
    return _runtime


@broker.task(
    task_name="airgap_rag.jobs.ingest_document",
    retry_on_error=True,
    max_retries=get_settings().job_max_attempts - 1,
    delay=get_settings().job_retry_delay_seconds,
    types_of_exceptions=(RetryableJobError,),
)
async def ingest_document_task(job_id: str) -> str:
    runtime = await get_worker_runtime()
    parsed_job_id = UUID(job_id)
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
        )
        return (await service.execute(parsed_job_id)).value


class TaskiqJobPublisher(JobPublisher):
    async def enqueue_ingestion(self, job_id: UUID) -> None:
        await ingest_document_task.kiq(str(job_id))


@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def close_worker_runtime(state: TaskiqState) -> None:
    del state
    global _runtime
    if _runtime is not None:
        await _runtime.close()
        _runtime = None
