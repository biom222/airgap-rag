from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: UUID | None = None
    document_ids: list[UUID] | None = None

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class ChatSourceResponse(BaseModel):
    document_id: UUID
    filename: str
    page: int | None
    chunk_id: UUID
    score: float
    rerank_score: float | None


class ChatResponse(BaseModel):
    answer: str
    session_id: UUID
    sources: list[ChatSourceResponse]
