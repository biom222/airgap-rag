from typing import Annotated

from fastapi import APIRouter, Depends, Request

from airgap_rag.api.dependencies.retrieval import get_retrieval_service
from airgap_rag.api.schemas.retrieval import (
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    RetrievedChunkResponse,
)
from airgap_rag.core.config import Settings
from airgap_rag.retrieval.service import RetrievalService

router = APIRouter(prefix="/api/v1/retrieval", tags=["retrieval"])


@router.post("/search", response_model=RetrievalSearchResponse)
async def search(
    payload: RetrievalSearchRequest,
    request: Request,
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> RetrievalSearchResponse:
    settings: Settings = request.app.state.settings
    results = await service.search(
        payload.question,
        top_k=payload.top_k or settings.retrieval_top_k,
        top_n=payload.top_n or settings.reranker_top_n,
        document_ids=payload.document_ids,
    )
    return RetrievalSearchResponse(
        results=[RetrievedChunkResponse.model_validate(result) for result in results]
    )
