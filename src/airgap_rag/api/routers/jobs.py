from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from airgap_rag.api.dependencies.jobs import get_job_query_service
from airgap_rag.api.schemas.jobs import JobResponse
from airgap_rag.jobs.service import JobQueryService

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: UUID,
    service: Annotated[JobQueryService, Depends(get_job_query_service)],
) -> JobResponse:
    job = await service.get(job_id)
    return JobResponse.model_validate(job)
