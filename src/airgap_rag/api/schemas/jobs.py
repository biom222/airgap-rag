from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from airgap_rag.jobs.types import JobStatus, JobType


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    type: JobType
    status: JobStatus
    progress: int
    attempt: int
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    updated_at: datetime
