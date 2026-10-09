import math
import statistics
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class MetricSummary(BaseModel):
    minimum: float
    mean: float
    p50: float
    p95: float
    maximum: float


def summarize(values: list[float]) -> MetricSummary:
    if not values:
        raise ValueError("at least one measurement is required")
    ordered = sorted(values)

    def percentile(value: float) -> float:
        index = max(0, math.ceil(value * len(ordered)) - 1)
        return ordered[index]

    return MetricSummary(
        minimum=ordered[0],
        mean=statistics.fmean(ordered),
        p50=percentile(0.50),
        p95=percentile(0.95),
        maximum=ordered[-1],
    )


class LLMSample(BaseModel):
    ttft_ms: float = Field(gt=0)
    tokens_per_second: float = Field(gt=0)
    total_latency_ms: float = Field(gt=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(gt=0)


class LLMReport(BaseModel):
    benchmark: str = "llm"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str
    runtime: str
    quantization: str
    hardware: dict[str, Any]
    model_size_mb: float | None
    ram_usage_mb: float | None
    vram_usage_mb: float | None
    warmup_iterations: int
    measured_iterations: int
    prompt_sha256: str
    samples: list[LLMSample]
    ttft_ms: MetricSummary
    tokens_per_second: MetricSummary
    total_latency_ms: MetricSummary


class RAGSample(BaseModel):
    embedding_latency_ms: float = Field(ge=0)
    retrieval_latency_ms: float = Field(ge=0)
    reranking_latency_ms: float = Field(ge=0)
    generation_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(gt=0)
    retrieved_chunks: int = Field(ge=0)


class RAGReport(BaseModel):
    benchmark: str = "rag"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    hardware: dict[str, Any]
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    reranker: str | None
    warmup_iterations: int
    measured_iterations: int
    question_sha256: str
    samples: list[RAGSample]
    embedding_latency_ms: MetricSummary
    retrieval_latency_ms: MetricSummary
    reranking_latency_ms: MetricSummary
    generation_latency_ms: MetricSummary
    total_latency_ms: MetricSummary
