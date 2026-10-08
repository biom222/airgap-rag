from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatHistoryMessage:
    role: ChatRole
    content: str


@dataclass(frozen=True, slots=True)
class Citation:
    document_id: UUID
    filename: str
    page: int | None
    chunk_id: UUID
    vector_score: float
    rerank_score: float | None


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    answer: str
    session_id: UUID
    sources: tuple[Citation, ...]


@dataclass(frozen=True, slots=True)
class RAGStreamToken:
    text: str


@dataclass(frozen=True, slots=True)
class RAGStreamDone:
    session_id: UUID


RAGStreamEvent = RAGStreamToken | RAGStreamDone


@dataclass(frozen=True, slots=True)
class RAGStream:
    sources: tuple[Citation, ...]
    events: AsyncIterator[RAGStreamEvent]
