import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from airgap_rag.api.dependencies.chat import get_chat_service
from airgap_rag.api.schemas.chat import (
    ChatDoneEvent,
    ChatErrorEvent,
    ChatRequest,
    ChatResponse,
    ChatSourceResponse,
    ChatSourcesEvent,
    ChatTokenEvent,
)
from airgap_rag.core.config import Settings
from airgap_rag.core.exceptions import AppError
from airgap_rag.rag.service import RAGService
from airgap_rag.rag.types import Citation, RAGStreamDone, RAGStreamToken

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
logger = logging.getLogger(__name__)


def _source_response(source: Citation) -> ChatSourceResponse:
    return ChatSourceResponse(
        document_id=source.document_id,
        filename=source.filename,
        page=source.page,
        chunk_id=source.chunk_id,
        score=source.vector_score,
        rerank_score=source.rerank_score,
    )


def _sse_event(name: str, payload: BaseModel) -> str:
    return f"event: {name}\ndata: {payload.model_dump_json()}\n\n"


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
        sources=[_source_response(source) for source in result.sources],
    )


@router.post(
    "/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream_chat(
    payload: ChatRequest,
    request: Request,
    service: Annotated[RAGService, Depends(get_chat_service)],
) -> StreamingResponse:
    settings: Settings = request.app.state.settings
    result = await service.stream_answer(
        payload.question,
        session_id=payload.session_id,
        document_ids=payload.document_ids,
        top_k=settings.retrieval_top_k,
        top_n=settings.reranker_top_n,
    )

    async def events() -> AsyncGenerator[str, None]:
        stream = result.events
        try:
            if await request.is_disconnected():
                return
            sources = ChatSourcesEvent(
                sources=[_source_response(source) for source in result.sources]
            )
            yield _sse_event("sources", sources)
            async for item in stream:
                if await request.is_disconnected():
                    break
                if isinstance(item, RAGStreamToken):
                    yield _sse_event("token", ChatTokenEvent(text=item.text))
                elif isinstance(item, RAGStreamDone):
                    yield _sse_event("done", ChatDoneEvent(session_id=item.session_id))
        except AppError as exc:
            if not await request.is_disconnected():
                yield _sse_event(
                    "error",
                    ChatErrorEvent(
                        code=exc.code,
                        message=exc.message,
                        request_id=str(request.state.request_id),
                    ),
                )
        except Exception as exc:  # Streaming boundary must log failures after HTTP 200.
            logger.error(
                "chat_stream_failed",
                extra={
                    "request_id": str(request.state.request_id),
                    "error_type": type(exc).__name__,
                },
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            if not await request.is_disconnected():
                yield _sse_event(
                    "error",
                    ChatErrorEvent(
                        code="internal_server_error",
                        message="Внутренняя ошибка сервера.",
                        request_id=str(request.state.request_id),
                    ),
                )
        finally:
            if isinstance(stream, AsyncGenerator):
                await stream.aclose()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
