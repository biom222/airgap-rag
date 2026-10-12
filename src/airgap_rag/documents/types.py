from dataclasses import dataclass
from datetime import date
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    department: str | None = None
    document_type: str | None = None
    document_date: date | None = None
    access_level: int = 0
    tags: tuple[str, ...] = ()


class DocumentStatus(StrEnum):
    """Lifecycle state persisted for a document."""

    PENDING = "PENDING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"
