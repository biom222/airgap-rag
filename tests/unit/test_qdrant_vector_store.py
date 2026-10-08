from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from airgap_rag.vector_store.base import VectorPoint, VectorStoreError
from airgap_rag.vector_store.qdrant import QdrantVectorStore

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


async def test_qdrant_store_upserts_filters_and_deletes() -> None:
    client = AsyncQdrantClient(location=":memory:")
    store = QdrantVectorStore(client, "test_chunks", timeout_seconds=5)
    first_document = uuid4()
    second_document = uuid4()
    first_chunk = uuid4()
    second_chunk = uuid4()
    try:
        await store.ensure_collection(3)
        await store.upsert(
            [
                VectorPoint(
                    id=first_chunk,
                    vector=[1.0, 0.0, 0.0],
                    payload={"document_id": str(first_document), "filename": "first.txt"},
                ),
                VectorPoint(
                    id=second_chunk,
                    vector=[0.0, 1.0, 0.0],
                    payload={"document_id": str(second_document), "filename": "second.txt"},
                ),
            ]
        )

        results = await store.search([1.0, 0.0, 0.0], limit=5, document_ids=[first_document])
        assert [result.id for result in results] == [first_chunk]

        await store.delete_document(first_document)
        remaining = await store.search([1.0, 0.0, 0.0], limit=5)
        assert [result.id for result in remaining] == [second_chunk]
    finally:
        await store.close()


async def test_qdrant_store_rejects_collection_dimension_mismatch() -> None:
    client = AsyncQdrantClient(location=":memory:")
    store = QdrantVectorStore(client, "test_chunks", timeout_seconds=5)
    try:
        await store.ensure_collection(3)
        with pytest.raises(VectorStoreError, match="expected 4"):
            await store.ensure_collection(4)
    finally:
        await store.close()
