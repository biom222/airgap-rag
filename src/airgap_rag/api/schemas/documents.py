from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from airgap_rag.documents.types import DocumentStatus


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    mime_type: str
    size: int
    sha256: str
    status: DocumentStatus
    page_count: int | None
    department: str | None
    document_type: str | None
    document_date: date | None
    access_level: int
    tags: list[str]
    created_at: datetime
    updated_at: datetime

    @field_validator("access_level", mode="before")
    @classmethod
    def default_access_level(cls, value: int | None) -> int:
        return 0 if value is None else value

    @field_validator("tags", mode="before")
    @classmethod
    def default_tags(cls, value: list[str] | None) -> list[str]:
        return [] if value is None else value


class DocumentUploadResponse(DocumentResponse):
    job_id: UUID
    deduplicated: bool
