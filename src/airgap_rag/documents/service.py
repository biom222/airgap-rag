from dataclasses import dataclass
from uuid import UUID, uuid4

from anyio import to_thread
from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError

from airgap_rag.db.models.documents import Document
from airgap_rag.db.models.jobs import Job
from airgap_rag.db.repositories.documents import DocumentRepository
from airgap_rag.documents.errors import DocumentNotFoundError
from airgap_rag.documents.storage import LocalDocumentStore
from airgap_rag.documents.types import DocumentStatus
from airgap_rag.documents.validation import DocumentValidator
from airgap_rag.jobs.errors import JobDispatchError
from airgap_rag.jobs.service import JobPublisher
from airgap_rag.jobs.types import JobStatus, JobType


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
        publisher: JobPublisher,
    ) -> None:
        self._repository = repository
        self._store = store
        self._validator = validator
        self._publisher = publisher

    async def upload(self, upload: UploadFile) -> UploadResult:
        staged = await self._store.stage(upload)
        promoted_storage_key: str | None = None
        try:
            validated = await to_thread.run_sync(
                self._validator.validate,
                staged.path,
                upload.filename,
                upload.content_type,
            )
            existing = await self._repository.find_by_sha256(staged.sha256)
            if existing is not None:
                await self._store.discard(staged.path)
                return await self._existing_upload(existing)

            document_id = uuid4()
            storage_key = self._store.make_storage_key(document_id, staged.sha256, validated.suffix)
            document = Document(
                id=document_id,
                filename=validated.filename,
                storage_key=storage_key,
                mime_type=validated.mime_type,
                size=staged.size,
                sha256=staged.sha256,
                status=DocumentStatus.PENDING,
                page_count=None,
            )
            job = Job(
                id=uuid4(),
                document_id=document.id,
                type=JobType.INGESTION,
                status=JobStatus.PENDING,
                progress=0,
                attempt=0,
            )
            await self._store.promote(staged.path, storage_key)
            promoted_storage_key = storage_key
            try:
                await self._repository.save(document, job)
            except IntegrityError:
                await self._repository.rollback()
                await self._store.delete(storage_key)
                promoted_storage_key = None
                concurrent = await self._repository.find_by_sha256(staged.sha256)
                if concurrent is None:
                    raise
                return await self._existing_upload(concurrent)
            except Exception:
                await self._repository.rollback()
                await self._store.delete(storage_key)
                promoted_storage_key = None
                raise
            promoted_storage_key = None
            await self._dispatch(job.id)
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
            await self._dispatch(job.id)
        return UploadResult(document=document, job=job, deduplicated=True)

    async def _dispatch(self, job_id: UUID) -> None:
        try:
            await self._publisher.enqueue_ingestion(job_id)
        except Exception as error:
            raise JobDispatchError() from error

    async def get(self, document_id: UUID) -> Document:
        document = await self._repository.get(document_id)
        if document is None:
            raise DocumentNotFoundError()
        return document
