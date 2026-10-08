from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import UploadFile
from httpx import ASGITransport, AsyncClient

from airgap_rag.api.dependencies.documents import get_document_service
from airgap_rag.core.config import Settings
from airgap_rag.db.models.documents import Document
from airgap_rag.db.models.jobs import Job
from airgap_rag.documents.service import DocumentService, UploadResult
from airgap_rag.documents.types import DocumentStatus
from airgap_rag.embeddings.mock import MockEmbeddingProvider
from airgap_rag.jobs.types import JobStatus, JobType
from airgap_rag.main import create_app
from tests.conftest import FakeDatabase, FakeJobPublisher, FakeVectorStore


class StubDocumentService(DocumentService):
    def __init__(self, document: Document, job: Job, deduplicated: bool) -> None:
        self.document = document
        self.job = job
        self.deduplicated = deduplicated

    async def upload(self, upload: UploadFile) -> UploadResult:
        return UploadResult(self.document, self.job, self.deduplicated)

    async def get(self, document_id: UUID) -> Document:
        return self.document


def make_document() -> Document:
    now = datetime.now(UTC)
    return Document(
        id=uuid4(),
        filename="notes.txt",
        storage_key="documents/id/hash.txt",
        mime_type="text/plain",
        size=5,
        sha256="a" * 64,
        status=DocumentStatus.PENDING,
        page_count=None,
        created_at=now,
        updated_at=now,
    )


def make_job(document: Document) -> Job:
    now = datetime.now(UTC)
    return Job(
        id=uuid4(),
        document_id=document.id,
        type=JobType.INGESTION,
        status=JobStatus.PENDING,
        progress=0,
        attempt=0,
        created_at=now,
        updated_at=now,
    )


async def make_client(
    settings: Settings, service: StubDocumentService
) -> AsyncIterator[AsyncClient]:
    application = create_app(
        settings=settings,
        database=FakeDatabase(),
        embedding_provider=MockEmbeddingProvider(8),
        vector_store=FakeVectorStore(),
        job_publisher=FakeJobPublisher(),
    )

    async def override_service() -> AsyncIterator[DocumentService]:
        yield service

    application.dependency_overrides[get_document_service] = override_service
    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application), base_url="http://test"
        ) as client:
            yield client


async def test_upload_returns_202_with_job_for_new_document(settings: Settings) -> None:
    document = make_document()
    job = make_job(document)
    service = StubDocumentService(document, job, deduplicated=False)

    async for client in make_client(settings, service):
        response = await client.post(
            "/api/v1/documents", files={"file": ("notes.txt", b"hello", "text/plain")}
        )

    assert response.status_code == 202
    assert response.json()["deduplicated"] is False
    assert response.json()["status"] == "PENDING"
    assert response.json()["job_id"] == str(job.id)


async def test_upload_returns_existing_document_for_duplicate(settings: Settings) -> None:
    document = make_document()
    service = StubDocumentService(document, make_job(document), deduplicated=True)

    async for client in make_client(settings, service):
        response = await client.post(
            "/api/v1/documents", files={"file": ("copy.txt", b"hello", "text/plain")}
        )

    assert response.status_code == 200
    assert response.json()["deduplicated"] is True


async def test_get_document_returns_metadata(settings: Settings) -> None:
    document = make_document()
    service = StubDocumentService(document, make_job(document), deduplicated=False)

    async for client in make_client(settings, service):
        response = await client.get(f"/api/v1/documents/{document.id}")

    assert response.status_code == 200
    assert response.json()["id"] == str(document.id)
    assert response.json()["sha256"] == "a" * 64
