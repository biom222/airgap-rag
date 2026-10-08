from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from anyio import to_thread
from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError
from starlette.datastructures import Headers

from airgap_rag.db.models.documents import Document
from airgap_rag.documents.errors import DocumentTooLargeError
from airgap_rag.documents.service import DocumentService
from airgap_rag.documents.storage import LocalDocumentStore
from airgap_rag.documents.types import DocumentStatus
from airgap_rag.documents.validation import DocumentValidator


class FakeDocumentRepository:
    def __init__(self) -> None:
        self.documents: dict[UUID, Document] = {}
        self.rollbacks = 0

    async def find_by_sha256(self, sha256: str) -> Document | None:
        return next(
            (document for document in self.documents.values() if document.sha256 == sha256), None
        )

    async def get(self, document_id: UUID) -> Document | None:
        return self.documents.get(document_id)

    async def save(self, document: Document) -> None:
        self.documents[document.id] = document

    async def rollback(self) -> None:
        self.rollbacks += 1


class RacingDocumentRepository(FakeDocumentRepository):
    async def save(self, document: Document) -> None:
        concurrent = Document(
            id=uuid4(),
            filename="concurrent.txt",
            storage_key="documents/concurrent/file.txt",
            mime_type=document.mime_type,
            size=document.size,
            sha256=document.sha256,
            status=DocumentStatus.PENDING,
            page_count=None,
        )
        self.documents[concurrent.id] = concurrent
        raise IntegrityError("INSERT", {}, RuntimeError("unique violation"))


def make_upload(filename: str, content: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


def make_service(runtime_path: Path, repository: FakeDocumentRepository) -> DocumentService:
    return DocumentService(
        repository=repository,
        store=LocalDocumentStore(runtime_path, max_upload_bytes=1024, read_chunk_bytes=4),
        validator=DocumentValidator(max_docx_uncompressed_bytes=4096),
    )


async def test_upload_stores_server_generated_file(runtime_path: Path) -> None:
    repository = FakeDocumentRepository()
    service = make_service(runtime_path, repository)

    result = await service.upload(make_upload("notes.txt", b"hello", "text/plain"))

    assert result.deduplicated is False
    assert result.document.filename == "notes.txt"
    assert result.document.sha256 == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    assert (runtime_path / result.document.storage_key).read_bytes() == b"hello"


async def test_duplicate_upload_reuses_document(runtime_path: Path) -> None:
    repository = FakeDocumentRepository()
    service = make_service(runtime_path, repository)

    first = await service.upload(make_upload("first.txt", b"same", "text/plain"))
    second = await service.upload(make_upload("second.txt", b"same", "text/plain"))

    assert second.deduplicated is True
    assert second.document.id == first.document.id
    assert len(repository.documents) == 1
    assert len(list((runtime_path / "documents").rglob("*.txt"))) == 1


async def test_concurrent_duplicate_removes_losing_file(runtime_path: Path) -> None:
    repository = RacingDocumentRepository()
    service = make_service(runtime_path, repository)

    result = await service.upload(make_upload("notes.txt", b"same", "text/plain"))

    assert result.deduplicated is True
    assert result.document.filename == "concurrent.txt"
    assert repository.rollbacks == 1
    assert list((runtime_path / "documents").rglob("*.txt")) == []


async def test_oversized_upload_cleans_staging(runtime_path: Path) -> None:
    repository = FakeDocumentRepository()
    service = DocumentService(
        repository=repository,
        store=LocalDocumentStore(runtime_path, max_upload_bytes=4, read_chunk_bytes=2),
        validator=DocumentValidator(max_docx_uncompressed_bytes=4096),
    )

    with pytest.raises(DocumentTooLargeError):
        await service.upload(make_upload("notes.txt", b"too large", "text/plain"))

    staged_files = await to_thread.run_sync(lambda: list(runtime_path.rglob("*.upload")))
    assert staged_files == []
    assert repository.documents == {}
