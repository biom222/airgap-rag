"""add model benchmark results

Revision ID: 20261009_0004
Revises: 20261008_0003
Create Date: 2026-10-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20261009_0004"
down_revision: str | None = "20261008_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_benchmarks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("runtime", sa.String(length=100), nullable=False),
        sa.Column("quantization", sa.String(length=100), nullable=False),
        sa.Column("hardware", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("model_size_mb", sa.Float(), nullable=True),
        sa.Column("ttft_ms", sa.Float(), nullable=False),
        sa.Column("tokens_per_second", sa.Float(), nullable=False),
        sa.Column("total_latency_ms", sa.Float(), nullable=False),
        sa.Column("ram_usage_mb", sa.Float(), nullable=True),
        sa.Column("vram_usage_mb", sa.Float(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_model_benchmarks_model_created",
        "model_benchmarks",
        ["model_name", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_model_benchmarks_model_created", table_name="model_benchmarks")
    op.drop_table("model_benchmarks")
