from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID


@dataclass(frozen=True, slots=True)
class VectorPoint:
    id: UUID
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    id: UUID
    score: float
    payload: dict[str, Any]


class VectorStore(Protocol):
    async def ensure_collection(self, dimension: int) -> None: ...

    async def delete_document(self, document_id: UUID) -> None: ...

    async def upsert(self, points: list[VectorPoint]) -> None: ...

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        document_ids: list[UUID] | None = None,
    ) -> list[VectorSearchResult]: ...

    async def healthcheck(self) -> bool: ...

    async def close(self) -> None: ...


class VectorStoreError(RuntimeError):
    """Vector storage operation failed or is misconfigured."""
