from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True, slots=True)
class RerankDocument:
    id: UUID
    text: str


@dataclass(frozen=True, slots=True)
class RerankResult:
    id: UUID
    score: float


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        documents: Sequence[RerankDocument],
    ) -> list[RerankResult]: ...

    async def healthcheck(self) -> bool: ...


class RerankerError(RuntimeError):
    """The configured local reranker could not score retrieved documents."""
