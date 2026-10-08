from typing import Annotated

from fastapi import APIRouter, Depends, Request

from airgap_rag.api.dependencies.chat import get_chat_service
from airgap_rag.api.schemas.chat import ChatRequest, ChatResponse, ChatSourceResponse
from airgap_rag.core.config import Settings
from airgap_rag.rag.service import RAGService

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    service: Annotated[RAGService, Depends(get_chat_service)],
) -> ChatResponse:
    settings: Settings = request.app.state.settings
    result = await service.answer(
        payload.question,
        session_id=payload.session_id,
        document_ids=payload.document_ids,
        top_k=settings.retrieval_top_k,
        top_n=settings.reranker_top_n,
    )
    return ChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        sources=[
            ChatSourceResponse(
                document_id=source.document_id,
                filename=source.filename,
                page=source.page,
                chunk_id=source.chunk_id,
                score=source.vector_score,
                rerank_score=source.rerank_score,
            )
            for source in result.sources
        ],
    )
