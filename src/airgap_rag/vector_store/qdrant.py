from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from airgap_rag.vector_store.base import (
    VectorPoint,
    VectorSearchResult,
    VectorStoreError,
)


class QdrantVectorStore:
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        timeout_seconds: int,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._timeout_seconds = timeout_seconds
        self._dimension: int | None = None

    async def ensure_collection(self, dimension: int) -> None:
        if self._dimension is not None:
            if self._dimension != dimension:
                raise VectorStoreError(
                    f"Qdrant collection dimension is {self._dimension}, expected {dimension}"
                )
            return

        exists = await self._client.collection_exists(self._collection_name)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
                timeout=self._timeout_seconds,
            )

        information = await self._client.get_collection(self._collection_name)
        vectors = information.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            raise VectorStoreError("Named vectors are not supported by this adapter")
        if vectors.size != dimension:
            raise VectorStoreError(
                f"Qdrant collection dimension is {vectors.size}, expected {dimension}"
            )
        if vectors.distance != models.Distance.COSINE:
            raise VectorStoreError("Qdrant collection must use cosine distance")

        await self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.UUID,
            wait=True,
        )
        self._dimension = dimension

    async def delete_document(self, document_id: UUID) -> None:
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def upsert(self, points: list[VectorPoint]) -> None:
        if not points:
            return
        if self._dimension is None:
            raise VectorStoreError("Qdrant collection was not initialized")
        if any(len(point.vector) != self._dimension for point in points):
            raise VectorStoreError("Vector dimension does not match the collection")

        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=str(point.id),
                    vector=point.vector,
                    payload=point.payload,
                )
                for point in points
            ],
            wait=True,
        )

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        document_ids: list[UUID] | None = None,
    ) -> list[VectorSearchResult]:
        query_filter = None
        if document_ids:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=[str(item) for item in document_ids]),
                    )
                ]
            )

        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            timeout=self._timeout_seconds,
        )
        results: list[VectorSearchResult] = []
        for point in response.points:
            try:
                point_id = UUID(str(point.id))
            except ValueError as exc:
                raise VectorStoreError("Qdrant returned a non-UUID point ID") from exc
            payload: dict[str, Any] = dict(point.payload or {})
            results.append(
                VectorSearchResult(
                    id=point_id,
                    score=float(point.score),
                    payload=payload,
                )
            )
        return results

    async def healthcheck(self) -> bool:
        try:
            await self._client.get_collections()
        except Exception:  # Infrastructure boundary: readiness returns a boolean.
            return False
        return True

    async def close(self) -> None:
        await self._client.close()
