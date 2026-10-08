import logging
from dataclasses import dataclass
from uuid import UUID

from anyio import to_thread

from airgap_rag.chunking.base import Chunk, Chunker
from airgap_rag.db.models.documents import Document
from airgap_rag.db.repositories.indexing import IndexingRepository
from airgap_rag.documents.errors import DocumentNotFoundError
from airgap_rag.documents.parsers import ParserRegistry
from airgap_rag.documents.storage import LocalDocumentStore
from airgap_rag.documents.types import DocumentStatus
from airgap_rag.embeddings.base import EmbeddingProvider, EmbeddingProviderError
from airgap_rag.vector_store.base import VectorPoint, VectorStore

logger = logging.getLogger(__name__)


class DocumentIndexingError(RuntimeError):
    """A document cannot be transformed into a valid vector index."""


@dataclass(frozen=True, slots=True)
class IndexingResult:
    document_id: UUID
    chunk_count: int
    status: DocumentStatus


class IndexingService:
    def __init__(
        self,
        repository: IndexingRepository,
        store: LocalDocumentStore,
        parser_registry: ParserRegistry,
        chunker: Chunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._repository = repository
        self._store = store
        self._parser_registry = parser_registry
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def index_document(
        self,
        document_id: UUID,
        *,
        mark_failed_on_error: bool = True,
    ) -> IndexingResult:
        document = await self._repository.get_document(document_id)
        if document is None:
            raise DocumentNotFoundError()

        try:
            chunks = await self._parse_and_chunk(document)
            await self._repository.replace_chunks(document.id, chunks)

            await self._repository.set_status(document, DocumentStatus.EMBEDDING)
            vectors = await self._embedding_provider.embed_documents(
                [chunk.text for chunk in chunks]
            )
            self._validate_vectors(chunks, vectors)

            await self._repository.set_status(document, DocumentStatus.INDEXING)
            await self._vector_store.ensure_collection(self._embedding_provider.dimension)
            await self._vector_store.delete_document(document.id)
            await self._vector_store.upsert(
                [
                    VectorPoint(
                        id=chunk.id,
                        vector=vector,
                        payload={
                            "document_id": str(document.id),
                            "chunk_id": str(chunk.id),
                            "filename": document.filename,
                            "page": chunk.page,
                            "chunk_index": chunk.chunk_index,
                            "text_hash": chunk.text_hash,
                        },
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ]
            )
            await self._repository.set_status(document, DocumentStatus.READY)
        except Exception:
            if mark_failed_on_error:
                await self._mark_failed(document)
            raise

        return IndexingResult(
            document_id=document.id,
            chunk_count=len(chunks),
            status=DocumentStatus.READY,
        )

    async def _parse_and_chunk(self, document: Document) -> tuple[Chunk, ...]:
        await self._repository.set_status(document, DocumentStatus.PARSING)
        parser = self._parser_registry.get(document.mime_type)
        path = self._store.resolve(document.storage_key)
        parsed = await to_thread.run_sync(parser.parse, path)

        page_count = (
            len(parsed.pages) if all(page.page is not None for page in parsed.pages) else None
        )
        await self._repository.set_status(
            document,
            DocumentStatus.CHUNKING,
            page_count=page_count,
        )
        chunks = self._chunker.split(document.id, parsed)
        if not chunks:
            raise DocumentIndexingError("Document does not contain extractable text")
        return chunks

    def _validate_vectors(self, chunks: tuple[Chunk, ...], vectors: list[list[float]]) -> None:
        if len(vectors) != len(chunks):
            raise EmbeddingProviderError("Embedding count does not match chunk count")
        if any(len(vector) != self._embedding_provider.dimension for vector in vectors):
            raise EmbeddingProviderError("Embedding dimension does not match provider dimension")

    async def _mark_failed(self, document: Document) -> None:
        try:
            await self._repository.rollback()
            await self._repository.set_status(document, DocumentStatus.FAILED)
        except Exception as status_error:  # Preserve the original indexing failure.
            logger.error(
                "failed_to_mark_document_failed",
                extra={
                    "document_id": str(document.id),
                    "error_type": type(status_error).__name__,
                },
            )
