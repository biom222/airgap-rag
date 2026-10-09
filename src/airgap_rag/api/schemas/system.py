from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, Literal["ok", "unavailable"]]


class HardwareResponse(BaseModel):
    python_version: str
    os: str
    os_version: str
    architecture: str
    cpu: str
    logical_cpu_count: int
    ram_total_mb: int
    gpu: str | None
    vram_total_mb: int | None
    cuda_available: bool


class SystemInfoResponse(BaseModel):
    hardware: HardwareResponse
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    reranker: str | None
    airgap_mode: bool


class ErrorDetails(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetails
