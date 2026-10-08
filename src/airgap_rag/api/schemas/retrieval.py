from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievalSearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[UUID] | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class RetrievedChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    filename: str
    page: int | None
    chunk_id: UUID
    chunk_index: int
    text: str
    vector_score: float


class RetrievalSearchResponse(BaseModel):
    results: list[RetrievedChunkResponse]
