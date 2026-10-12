from dataclasses import dataclass
from uuid import UUID, uuid4

from anyio import to_thread
from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError

from airgap_rag.db.models.documents import Document
from airgap_rag.db.models.jobs import Job
from airgap_rag.db.models.outbox import OutboxEvent
from airgap_rag.db.repositories.documents import DocumentRepository
from airgap_rag.documents.errors import DocumentNotFoundError
from airgap_rag.documents.storage import LocalDocumentStore
from airgap_rag.documents.types import DocumentMetadata, DocumentStatus
from airgap_rag.documents.validation import DocumentValidator
from airgap_rag.jobs.types import JobStatus, JobType
from airgap_rag.metrics import MetricsRecorder, NoOpMetrics


@dataclass(frozen=True, slots=True)
class UploadResult:
    document: Document
    job: Job
    deduplicated: bool


class DocumentService:
    def __init__(
        self,
        repository: DocumentRepository,
        store: LocalDocumentStore,
        validator: DocumentValidator,
        metrics: MetricsRecorder | None = None,
        owner_id: str = "local-development",
        can_manage_all: bool = False,
        trace_id: str | None = None,
        trace_context: dict[str, str] | None = None,
    ) -> None:
        self._repository = repository
        self._store = store
        self._validator = validator
        self._metrics = metrics or NoOpMetrics()
        self._owner_id = owner_id
        self._can_manage_all = can_manage_all
        self._trace_id = trace_id
        self._trace_context = trace_context or {}

    async def upload(
        self,
        upload: UploadFile,
        metadata: DocumentMetadata | None = None,
    ) -> UploadResult:
        resolved_metadata = metadata or DocumentMetadata()
        staged = await self._store.stage(upload)
        promoted_storage_key: str | None = None
        try:
            validated = await to_thread.run_sync(
                self._validator.validate,
                staged.path,
                upload.filename,
                upload.content_type,
            )
            existing = await self._repository.find_by_sha256(self._owner_id, staged.sha256)
            if existing is not None:
                await self._store.discard(staged.path)
                return await self._existing_upload(existing)

            document_id = uuid4()
            storage_key = self._store.make_storage_key(document_id, staged.sha256, validated.suffix)
            document = Document(
                id=document_id,
                owner_id=self._owner_id,
                filename=validated.filename,
                storage_key=storage_key,
                mime_type=validated.mime_type,
                size=staged.size,
                sha256=staged.sha256,
                status=DocumentStatus.PENDING,
                page_count=None,
                department=resolved_metadata.department,
                document_type=resolved_metadata.document_type,
                document_date=resolved_metadata.document_date,
                access_level=resolved_metadata.access_level,
                tags=list(resolved_metadata.tags),
            )
            job = Job(
                id=uuid4(),
                document_id=document.id,
                type=JobType.INGESTION,
                status=JobStatus.PENDING,
                progress=0,
                attempt=0,
                trace_id=self._trace_id,
            )
            event = OutboxEvent(
                id=uuid4(),
                event_type="ingestion.requested",
                aggregate_id=job.id,
                payload={
                    "job_id": str(job.id),
                    "trace_id": self._trace_id or "",
                    "trace_context": self._trace_context,
                },
                attempts=0,
            )
            await self._store.promote(staged.path, storage_key)
            promoted_storage_key = storage_key
            try:
                await self._repository.save(document, job, event)
            except IntegrityError:
                await self._repository.rollback()
                await self._store.delete(storage_key)
                promoted_storage_key = None
                concurrent = await self._repository.find_by_sha256(self._owner_id, staged.sha256)
                if concurrent is None:
                    raise
                return await self._existing_upload(concurrent)
            except Exception:
                await self._repository.rollback()
                await self._store.delete(storage_key)
                promoted_storage_key = None
                raise
            promoted_storage_key = None
            self._metrics.record_document_created()
            return UploadResult(document=document, job=job, deduplicated=False)
        except Exception:
            await self._store.discard(staged.path)
            if promoted_storage_key is not None:
                await self._store.delete(promoted_storage_key)
            raise

    async def _existing_upload(self, document: Document) -> UploadResult:
        job = await self._repository.find_latest_job(document.id)
        if job is None:
            raise RuntimeError(f"Document {document.id} has no ingestion job")
        if job.status == JobStatus.PENDING:
            await self._repository.ensure_outbox(
                job,
                trace_id=self._trace_id,
                trace_context=self._trace_context,
            )
        return UploadResult(document=document, job=job, deduplicated=True)

    async def get(self, document_id: UUID) -> Document:
        owner_scope = None if self._can_manage_all else self._owner_id
        document = await self._repository.get(document_id, owner_scope)
        if document is None:
            raise DocumentNotFoundError()
        return document
