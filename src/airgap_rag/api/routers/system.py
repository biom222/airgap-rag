import asyncio
import logging
from dataclasses import asdict
from typing import Protocol, cast

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from airgap_rag.api.schemas.system import HealthResponse, ReadinessResponse, SystemInfoResponse
from airgap_rag.core.config import Settings
from airgap_rag.system_info import collect_hardware_info

router = APIRouter(tags=["system"])
logger = logging.getLogger(__name__)


class PingableDatabase(Protocol):
    async def ping(self) -> None: ...


class HealthcheckDependency(Protocol):
    async def healthcheck(self) -> bool: ...


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/system/info", response_model=SystemInfoResponse)
async def system_info(request: Request) -> SystemInfoResponse:
    settings = cast(Settings, request.app.state.settings)
    hardware = collect_hardware_info()
    embedding_model = (
        settings.embedding_model_name_or_path
        if settings.embedding_provider == "sentence_transformer"
        else f"mock:{settings.embedding_dimension}"
    )
    reranker = None
    if settings.reranker_enabled:
        reranker = (
            settings.reranker_model_name_or_path
            if settings.reranker_provider == "cross_encoder"
            else "mock"
        )
    return SystemInfoResponse(
        hardware=asdict(hardware),
        llm_provider=settings.llm_provider,
        llm_model=settings.ollama_model if settings.llm_provider == "ollama" else "mock",
        embedding_provider=settings.embedding_provider,
        embedding_model=embedding_model,
        reranker=reranker,
        airgap_mode=settings.airgap_mode,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def ready(request: Request) -> ReadinessResponse | JSONResponse:
    database = cast(PingableDatabase, request.app.state.database)
    embedding_provider = cast(HealthcheckDependency, request.app.state.embedding_provider)
    vector_store = cast(HealthcheckDependency, request.app.state.vector_store)
    llm_provider = cast(HealthcheckDependency, request.app.state.llm_provider)
    reranker = cast(HealthcheckDependency | None, request.app.state.reranker)

    async def database_healthcheck() -> bool:
        try:
            await database.ping()
        except Exception as exc:  # Boundary: readiness reports dependency state.
            logger.warning("database_readiness_failed", extra={"error_type": type(exc).__name__})
            return False
        return True

    try:
        database_ok, embeddings_ok, qdrant_ok, llm_ok = await asyncio.gather(
            database_healthcheck(),
            embedding_provider.healthcheck(),
            vector_store.healthcheck(),
            llm_provider.healthcheck(),
        )
    except Exception as exc:  # Defensive boundary for provider contract violations.
        logger.error("readiness_check_failed", extra={"error_type": type(exc).__name__})
        database_ok, embeddings_ok, qdrant_ok, llm_ok = False, False, False, False

    checks = {
        "database": "ok" if database_ok else "unavailable",
        "embeddings": "ok" if embeddings_ok else "unavailable",
        "qdrant": "ok" if qdrant_ok else "unavailable",
        "llm": "ok" if llm_ok else "unavailable",
    }
    reranker_ok = True
    if reranker is not None:
        try:
            reranker_ok = await reranker.healthcheck()
        except Exception as exc:  # Defensive boundary for provider contract violations.
            logger.error("reranker_readiness_failed", extra={"error_type": type(exc).__name__})
            reranker_ok = False
        checks["reranker"] = "ok" if reranker_ok else "unavailable"

    if not all((database_ok, embeddings_ok, qdrant_ok, llm_ok, reranker_ok)):
        payload = ReadinessResponse(status="not_ready", checks=checks)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(),
        )

    return ReadinessResponse(status="ready", checks=checks)
