from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=4000)
    relevant_chunk_ids: tuple[UUID, ...] = Field(min_length=1)
    document_ids: tuple[UUID, ...] | None = None

    @field_validator("case_id", "question")
    @classmethod
    def strip_non_empty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("relevant_chunk_ids")
    @classmethod
    def reject_duplicate_relevant_chunks(cls, value: tuple[UUID, ...]) -> tuple[UUID, ...]:
        if len(value) != len(set(value)):
            raise ValueError("relevant_chunk_ids must not contain duplicates")
        return value

    @field_validator("document_ids")
    @classmethod
    def reject_empty_or_duplicate_document_filter(
        cls,
        value: tuple[UUID, ...] | None,
    ) -> tuple[UUID, ...] | None:
        if value is not None and not value:
            raise ValueError("document_ids must be null or non-empty")
        if value is not None and len(value) != len(set(value)):
            raise ValueError("document_ids must not contain duplicates")
        return value


class EvaluationCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    question: str
    relevant_chunk_ids: tuple[UUID, ...]
    retrieved_chunk_ids: tuple[UUID, ...]
    first_relevant_rank: int | None
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    hits_at_k: dict[int, bool]


class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    evaluated_at: datetime
    dataset_name: str
    base_url: str
    case_count: int = Field(ge=1)
    top_k: tuple[int, ...]
    hit_at_k: dict[int, float]
    mrr: float = Field(ge=0.0, le=1.0)
    mrr_cutoff: int = Field(ge=1)
    cases: tuple[EvaluationCaseResult, ...]
