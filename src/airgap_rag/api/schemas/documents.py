from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

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
    created_at: datetime
    updated_at: datetime


class DocumentUploadResponse(DocumentResponse):
    job_id: UUID
    deduplicated: bool
