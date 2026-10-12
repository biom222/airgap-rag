import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import Protocol
from uuid import UUID

from qdrant_client.http.exceptions import ResponseHandlingException

from airgap_rag.db.models.jobs import Job
from airgap_rag.db.repositories.jobs import JobRepository
from airgap_rag.documents.errors import DocumentParseError
from airgap_rag.embeddings.base import EmbeddingProviderError
from airgap_rag.indexing.service import DocumentIndexingError, IndexingService
from airgap_rag.jobs.errors import JobNotFoundError, RetryableJobError
from airgap_rag.metrics import MetricsRecorder, NoOpMetrics

logger = logging.getLogger(__name__)


class JobPublisher(Protocol):
    async def enqueue_ingestion(
        self,
        job_id: UUID,
        *,
        trace_context: dict[str, str] | None = None,
    ) -> None: ...


class JobExecutionOutcome(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class FailureDetails:
    code: str
    message: str
    retryable: bool


class JobQueryService:
    def __init__(self, repository: JobRepository, owner_id: str | None = None) -> None:
        self._repository = repository
        self._owner_id = owner_id

    async def get(self, job_id: UUID) -> Job:
        job = await self._repository.get_for_owner(job_id, self._owner_id)
        if job is None:
            raise JobNotFoundError()
        return job


class JobExecutionService:
    def __init__(
        self,
        repository: JobRepository,
        indexing_service: IndexingService,
        *,
        max_attempts: int,
        timeout_seconds: int,
        stale_after_seconds: int,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self._repository = repository
        self._indexing_service = indexing_service
        self._max_attempts = max_attempts
        self._timeout_seconds = timeout_seconds
        self._stale_after_seconds = stale_after_seconds
        self._metrics = metrics or NoOpMetrics()

    async def execute(self, job_id: UUID) -> JobExecutionOutcome:
        claimed = await self._repository.claim(
            job_id,
            stale_after_seconds=self._stale_after_seconds,
        )
        if claimed is None:
            return JobExecutionOutcome.SKIPPED

        job_id = claimed.job.id
        document_id = claimed.document.id
        attempt = claimed.job.attempt
        started_at = perf_counter()
        logger.info(
            "ingestion_job_started",
            extra={
                "job_id": str(job_id),
                "document_id": str(document_id),
                "trace_id": claimed.job.trace_id,
                "attempt": attempt,
            },
        )
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await self._indexing_service.index_document(
                    claimed.document.id,
                    mark_failed_on_error=False,
                )
        except Exception as error:
            await self._repository.rollback()
            failure = classify_job_failure(error)
            self._metrics.record_task_failure(failure.retryable)
            message = _safe_error_message(failure.message)
            if failure.retryable and attempt < self._max_attempts:
                await self._repository.prepare_retry(
                    job_id,
                    error_code=failure.code,
                    error_message=message,
                )
                logger.warning(
                    "ingestion_job_retry_scheduled",
                    extra={
                        "job_id": str(job_id),
                        "document_id": str(document_id),
                        "trace_id": claimed.job.trace_id,
                        "attempt": attempt,
                        "error_code": failure.code,
                        "duration_ms": round((perf_counter() - started_at) * 1000, 3),
                    },
                )
                raise RetryableJobError(message) from error

            await self._repository.fail(
                job_id,
                error_code=failure.code,
                error_message=message,
            )
            self._metrics.record_ingestion_failure()
            logger.error(
                "ingestion_job_failed",
                extra={
                    "job_id": str(job_id),
                    "document_id": str(document_id),
                    "trace_id": claimed.job.trace_id,
                    "attempt": attempt,
                    "error_code": failure.code,
                    "retryable": failure.retryable,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 3),
                },
            )
            return JobExecutionOutcome.FAILED

        logger.info(
            "ingestion_job_completed",
            extra={
                "job_id": str(job_id),
                "document_id": str(document_id),
                "trace_id": claimed.job.trace_id,
                "attempt": attempt,
                "duration_ms": round((perf_counter() - started_at) * 1000, 3),
            },
        )
        return JobExecutionOutcome.COMPLETED


def classify_job_failure(error: Exception) -> FailureDetails:
    if isinstance(error, TimeoutError):
        return FailureDetails("job_timeout", "Время выполнения задания истекло.", True)
    if isinstance(error, FileNotFoundError):
        return FailureDetails("source_file_missing", "Исходный файл документа не найден.", False)
    if isinstance(error, DocumentParseError):
        return FailureDetails("document_parse_failed", str(error), False)
    if isinstance(error, DocumentIndexingError):
        return FailureDetails("document_indexing_failed", str(error), False)
    if isinstance(error, EmbeddingProviderError):
        return FailureDetails("embedding_failed", str(error), False)
    if isinstance(error, ResponseHandlingException):
        return FailureDetails("qdrant_unavailable", str(error), True)
    if isinstance(error, (ConnectionError, OSError)):
        return FailureDetails("infrastructure_unavailable", str(error), True)
    return FailureDetails("temporary_indexing_failure", str(error), True)


def _safe_error_message(message: str, *, max_length: int = 2000) -> str:
    normalized = message.strip() or "Неизвестная ошибка фонового задания."
    return normalized[:max_length]
