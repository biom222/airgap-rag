from datetime import date
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from airgap_rag.retrieval.filters import RetrievalFilters


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: UUID | None = None
    document_ids: list[UUID] | None = None
    department: str | None = Field(default=None, min_length=1, max_length=100)
    document_type: str | None = Field(default=None, min_length=1, max_length=100)
    date_from: date | None = None
    date_to: date | None = None
    tags: list[str] = Field(default_factory=list, max_length=20)

    def filters(self, *, max_access_level: int | None) -> RetrievalFilters | None:
        if not any(
            (
                self.department,
                self.document_type,
                self.date_from,
                self.date_to,
                self.tags,
                max_access_level is not None,
            )
        ):
            return None
        return RetrievalFilters(
            department=self.department,
            document_type=self.document_type,
            date_from=self.date_from,
            date_to=self.date_to,
            tags=tuple(self.tags),
            max_access_level=max_access_level,
        )

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(tag.strip() for tag in value if tag.strip()))
        if any(len(tag) > 100 for tag in normalized):
            raise ValueError("each tag must contain at most 100 characters")
        return normalized

    @model_validator(mode="after")
    def validate_date_range(self) -> "ChatRequest":
        if self.date_from is not None and self.date_to is not None:
            if self.date_from > self.date_to:
                raise ValueError("date_from must not be after date_to")
        return self


class ChatSourceResponse(BaseModel):
    document_id: UUID
    filename: str
    page: int | None
    chunk_id: UUID
    score: float
    lexical_score: float | None = Field(default=None, exclude_if=lambda value: value is None)
    fusion_score: float | None = Field(default=None, exclude_if=lambda value: value is None)
    rerank_score: float | None


class ChatResponse(BaseModel):
    answer: str
    session_id: UUID
    sources: list[ChatSourceResponse]


class ChatSourcesEvent(BaseModel):
    sources: list[ChatSourceResponse]


class ChatTokenEvent(BaseModel):
    text: str


class ChatDoneEvent(BaseModel):
    session_id: UUID


class ChatErrorEvent(BaseModel):
    code: str
    message: str
    request_id: str
