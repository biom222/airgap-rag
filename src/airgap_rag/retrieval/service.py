import logging
from dataclasses import dataclass
from uuid import UUID

from airgap_rag.db.repositories.indexing import IndexingRepository
from airgap_rag.embeddings.base import EmbeddingProvider
from airgap_rag.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    document_id: UUID
    filename: str
    page: int | None
    chunk_id: UUID
    chunk_index: int
    text: str
    vector_score: float


class RetrievalService:
    def __init__(
        self,
        repository: IndexingRepository,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def search(
        self,
        question: str,
        *,
        top_k: int,
        document_ids: list[UUID] | None = None,
    ) -> list[RetrievedChunk]:
        vector = await self._embedding_provider.embed_query(question)
        await self._vector_store.ensure_collection(self._embedding_provider.dimension)
        hits = await self._vector_store.search(
            vector,
            limit=top_k,
            document_ids=document_ids,
        )
        chunks = await self._repository.get_chunks([hit.id for hit in hits])

        results: list[RetrievedChunk] = []
        for hit in hits:
            chunk = chunks.get(hit.id)
            if chunk is None:
                logger.warning("retrieved_chunk_missing", extra={"chunk_id": str(hit.id)})
                continue
            filename = hit.payload.get("filename")
            if not isinstance(filename, str) or not filename:
                logger.warning("retrieved_payload_invalid", extra={"chunk_id": str(hit.id)})
                continue
            results.append(
                RetrievedChunk(
                    document_id=chunk.document_id,
                    filename=filename,
                    page=chunk.page,
                    chunk_id=chunk.id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    vector_score=hit.score,
                )
            )
        return results
