from pathlib import Path
from uuid import UUID, uuid4

import pytest

from airgap_rag.chunking.base import Chunk
from airgap_rag.chunking.recursive import RecursiveCharacterChunker
from airgap_rag.db.models.documents import Document, DocumentChunk
from airgap_rag.documents.parsers import ParserRegistry, TextParser
from airgap_rag.documents.storage import LocalDocumentStore
from airgap_rag.documents.types import DocumentStatus
from airgap_rag.embeddings.mock import MockEmbeddingProvider
from airgap_rag.indexing.service import IndexingService
from airgap_rag.vector_store.base import VectorPoint, VectorSearchResult


class FakeIndexingRepository:
    def __init__(self, document: Document) -> None:
        self.document = document
        self.chunks: tuple[Chunk, ...] = ()
        self.statuses: list[DocumentStatus] = []
        self.rollbacks = 0

    async def get_document(self, document_id: UUID) -> Document | None:
        return self.document if self.document.id == document_id else None

    async def set_status(
        self,
        document: Document,
        status: DocumentStatus,
        *,
        page_count: int | None = None,
    ) -> None:
        document.status = status
        if page_count is not None:
            document.page_count = page_count
        self.statuses.append(status)

    async def replace_chunks(self, document_id: UUID, chunks: tuple[Chunk, ...]) -> None:
        self.chunks = chunks

    async def get_chunks(self, chunk_ids: list[UUID]) -> dict[UUID, DocumentChunk]:
        return {}

    async def rollback(self) -> None:
        self.rollbacks += 1


class FakeIndexVectorStore:
    def __init__(
        self,
        fail_upsert: bool = False,
        partial_failures_remaining: int = 0,
    ) -> None:
        self.dimension: int | None = None
        self.points: dict[UUID, VectorPoint] = {}
        self.deleted_documents: list[UUID] = []
        self.fail_upsert = fail_upsert
        self.partial_failures_remaining = partial_failures_remaining

    async def ensure_collection(self, dimension: int) -> None:
        self.dimension = dimension

    async def delete_document(self, document_id: UUID) -> None:
        self.deleted_documents.append(document_id)
        self.points = {
            point_id: point
            for point_id, point in self.points.items()
            if point.payload["document_id"] != str(document_id)
        }

    async def upsert(self, points: list[VectorPoint]) -> None:
        if self.fail_upsert:
            raise RuntimeError("qdrant unavailable")
        if self.partial_failures_remaining > 0:
            self.partial_failures_remaining -= 1
            self.points[points[0].id] = points[0]
            raise ConnectionError("worker crashed after partial vector write")
        self.points.update({point.id: point for point in points})

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        document_ids: list[UUID] | None = None,
    ) -> list[VectorSearchResult]:
        return []

    async def healthcheck(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def write_stored_document(root: Path, storage_key: str, content: str) -> None:
    path = root / storage_key
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")


def make_document() -> Document:
    return Document(
        id=uuid4(),
        filename="notes.txt",
        storage_key="documents/source/notes.txt",
        mime_type="text/plain",
        size=64,
        sha256="a" * 64,
        status=DocumentStatus.PENDING,
        page_count=None,
    )


def make_service(
    root: Path,
    repository: FakeIndexingRepository,
    vector_store: FakeIndexVectorStore,
) -> IndexingService:
    return IndexingService(
        repository=repository,
        store=LocalDocumentStore(root, max_upload_bytes=1024, read_chunk_bytes=128),
        parser_registry=ParserRegistry((TextParser(),)),
        chunker=RecursiveCharacterChunker(chunk_size=24, chunk_overlap=4),
        embedding_provider=MockEmbeddingProvider(8),
        vector_store=vector_store,
    )


async def test_indexing_is_idempotent_for_repeated_run(runtime_path: Path) -> None:
    document = make_document()
    write_stored_document(runtime_path, document.storage_key, "alpha beta gamma delta epsilon")
    repository = FakeIndexingRepository(document)
    vector_store = FakeIndexVectorStore()
    service = make_service(runtime_path, repository, vector_store)

    first = await service.index_document(document.id)
    first_ids = set(vector_store.points)
    second = await service.index_document(document.id)

    assert first.status == DocumentStatus.READY
    assert second.chunk_count == first.chunk_count
    assert set(vector_store.points) == first_ids
    assert vector_store.deleted_documents == [document.id, document.id]
    assert all("text" not in point.payload for point in vector_store.points.values())
    assert document.status == DocumentStatus.READY


async def test_indexing_marks_document_failed_when_qdrant_fails(runtime_path: Path) -> None:
    document = make_document()
    write_stored_document(runtime_path, document.storage_key, "extractable text")
    repository = FakeIndexingRepository(document)
    vector_store = FakeIndexVectorStore(fail_upsert=True)
    service = make_service(runtime_path, repository, vector_store)

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        await service.index_document(document.id)

    assert document.status == DocumentStatus.FAILED
    assert repository.statuses[-1] == DocumentStatus.FAILED
    assert repository.chunks
    assert repository.rollbacks == 1


async def test_retry_replaces_partial_vectors_without_duplicates(runtime_path: Path) -> None:
    document = make_document()
    write_stored_document(
        runtime_path,
        document.storage_key,
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda",
    )
    repository = FakeIndexingRepository(document)
    vector_store = FakeIndexVectorStore(partial_failures_remaining=1)
    service = make_service(runtime_path, repository, vector_store)

    with pytest.raises(ConnectionError, match="partial vector write"):
        await service.index_document(document.id, mark_failed_on_error=False)

    partial_ids = set(vector_store.points)
    assert len(partial_ids) == 1

    result = await service.index_document(document.id, mark_failed_on_error=False)

    assert result.status == DocumentStatus.READY
    assert len(vector_store.points) == result.chunk_count
    assert partial_ids <= set(vector_store.points)
    assert vector_store.deleted_documents == [document.id, document.id]
