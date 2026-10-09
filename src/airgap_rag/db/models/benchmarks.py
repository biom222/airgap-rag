from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from airgap_rag.db.base import Base


class ModelBenchmark(Base):
    __tablename__ = "model_benchmarks"
    __table_args__ = (Index("ix_model_benchmarks_model_created", "model_name", "created_at"),)

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    runtime: Mapped[str] = mapped_column(String(100), nullable=False)
    quantization: Mapped[str] = mapped_column(String(100), nullable=False)
    hardware: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    model_size_mb: Mapped[float | None] = mapped_column(Float)
    ttft_ms: Mapped[float] = mapped_column(Float, nullable=False)
    tokens_per_second: Mapped[float] = mapped_column(Float, nullable=False)
    total_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    ram_usage_mb: Mapped[float | None] = mapped_column(Float)
    vram_usage_mb: Mapped[float | None] = mapped_column(Float)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
