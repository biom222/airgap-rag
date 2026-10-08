from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from airgap_rag.documents.parsers import ParsedDocument


@dataclass(frozen=True, slots=True)
class Chunk:
    id: UUID
    document_id: UUID
    chunk_index: int
    page: int | None
    text: str
    text_hash: str


class Chunker(Protocol):
    def split(self, document_id: UUID, document: ParsedDocument) -> tuple[Chunk, ...]: ...
