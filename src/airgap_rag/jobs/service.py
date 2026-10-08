import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from qdrant_client.http.exceptions import ResponseHandlingException

from airgap_rag.db.models.jobs import Job
from airgap_rag.db.repositories.jobs import JobRepository
from airgap_rag.documents.errors import DocumentParseError
from airgap_rag.embeddings.base import EmbeddingProviderError
from airgap_rag.indexing.service import DocumentIndexingError, IndexingService
from airgap_rag.jobs.errors import JobNotFoundError, RetryableJobError

logger = logging.getLogger(__name__)


class JobPublisher(Protocol):
    async def enqueue_ingestion(self, job_id: UUID) -> None: ...


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
    def __init__(self, repository: JobRepository) -> None:
        self._repository = repository

    async def get(self, job_id: UUID) -> Job:
        job = await self._repository.get(job_id)
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
    ) -> None:
        self._repository = repository
        self._indexing_service = indexing_service
        self._max_attempts = max_attempts
        self._timeout_seconds = timeout_seconds
        self._stale_after_seconds = stale_after_seconds

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
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await self._indexing_service.index_document(
                    claimed.document.id,
                    mark_failed_on_error=False,
                )
        except Exception as error:
            await self._repository.rollback()
            failure = classify_job_failure(error)
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
                        "attempt": attempt,
                        "error_code": failure.code,
                    },
                )
                raise RetryableJobError(message) from error

            await self._repository.fail(
                job_id,
                error_code=failure.code,
                error_message=message,
            )
            logger.error(
                "ingestion_job_failed",
                extra={
                    "job_id": str(job_id),
                    "document_id": str(document_id),
                    "attempt": attempt,
                    "error_code": failure.code,
                    "retryable": failure.retryable,
                },
            )
            return JobExecutionOutcome.FAILED

        logger.info(
            "ingestion_job_completed",
            extra={
                "job_id": str(job_id),
                "document_id": str(document_id),
                "attempt": attempt,
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
