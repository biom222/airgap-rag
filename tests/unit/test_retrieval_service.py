from collections.abc import Sequence
from uuid import UUID, uuid4

from airgap_rag.chunking.base import Chunk
from airgap_rag.db.models.documents import Document, DocumentChunk
from airgap_rag.documents.types import DocumentStatus
from airgap_rag.embeddings.mock import MockEmbeddingProvider
from airgap_rag.reranking.base import RerankDocument, RerankResult
from airgap_rag.retrieval.service import RetrievalService
from airgap_rag.vector_store.base import VectorPoint, VectorSearchResult


class FakeRetrievalRepository:
    def __init__(self, chunks: dict[UUID, DocumentChunk]) -> None:
        self.chunks = chunks

    async def get_document(self, document_id: UUID) -> Document | None:
        return None

    async def set_status(
        self,
        document: Document,
        status: DocumentStatus,
        *,
        page_count: int | None = None,
    ) -> None:
        return None

    async def replace_chunks(self, document_id: UUID, chunks: tuple[Chunk, ...]) -> None:
        return None

    async def get_chunks(self, chunk_ids: list[UUID]) -> dict[UUID, DocumentChunk]:
        return {
            chunk_id: self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks
        }

    async def rollback(self) -> None:
        return None


class FakeSearchVectorStore:
    def __init__(self, hits: list[VectorSearchResult]) -> None:
        self.hits = hits
        self.dimension: int | None = None
        self.document_ids: list[UUID] | None = None

    async def ensure_collection(self, dimension: int) -> None:
        self.dimension = dimension

    async def delete_document(self, document_id: UUID) -> None:
        return None

    async def upsert(self, points: list[VectorPoint]) -> None:
        return None

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        document_ids: list[UUID] | None = None,
    ) -> list[VectorSearchResult]:
        self.document_ids = document_ids
        return self.hits[:limit]

    async def healthcheck(self) -> bool:
        return True

    async def close(self) -> None:
        return None


async def test_retrieval_joins_qdrant_scores_with_postgres_text() -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    missing_chunk_id = uuid4()
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=document_id,
        chunk_index=2,
        page=4,
        text="Authoritative text from PostgreSQL",
        text_hash="b" * 64,
    )
    repository = FakeRetrievalRepository({chunk_id: chunk})
    vector_store = FakeSearchVectorStore(
        [
            VectorSearchResult(
                id=chunk_id,
                score=0.91,
                payload={"filename": "policy.pdf", "text": "untrusted payload text"},
            ),
            VectorSearchResult(
                id=missing_chunk_id,
                score=0.8,
                payload={"filename": "missing.pdf"},
            ),
        ]
    )
    service = RetrievalService(repository, MockEmbeddingProvider(8), vector_store)

    results = await service.search("retention period", top_k=5, document_ids=[document_id])

    assert len(results) == 1
    assert results[0].text == "Authoritative text from PostgreSQL"
    assert results[0].vector_score == 0.91
    assert results[0].page == 4
    assert vector_store.dimension == 8
    assert vector_store.document_ids == [document_id]


class ReverseReranker:
    async def rerank(
        self,
        query: str,
        documents: Sequence[RerankDocument],
    ) -> list[RerankResult]:
        del query
        return [
            RerankResult(id=document.id, score=float(index))
            for index, document in enumerate(documents)
        ]

    async def healthcheck(self) -> bool:
        return True


async def test_retrieval_preserves_vector_score_and_applies_reranking() -> None:
    document_id = uuid4()
    first_id = uuid4()
    second_id = uuid4()
    chunks = {
        first_id: DocumentChunk(
            id=first_id,
            document_id=document_id,
            chunk_index=0,
            page=1,
            text="first",
            text_hash="a" * 64,
        ),
        second_id: DocumentChunk(
            id=second_id,
            document_id=document_id,
            chunk_index=1,
            page=2,
            text="second",
            text_hash="b" * 64,
        ),
    }
    vector_store = FakeSearchVectorStore(
        [
            VectorSearchResult(id=first_id, score=0.9, payload={"filename": "doc.pdf"}),
            VectorSearchResult(id=second_id, score=0.8, payload={"filename": "doc.pdf"}),
        ]
    )
    service = RetrievalService(
        FakeRetrievalRepository(chunks),
        MockEmbeddingProvider(8),
        vector_store,
        ReverseReranker(),
    )

    results = await service.search("question", top_k=2, top_n=1)

    assert len(results) == 1
    assert results[0].chunk_id == second_id
    assert results[0].vector_score == 0.8
    assert results[0].rerank_score == 1.0
