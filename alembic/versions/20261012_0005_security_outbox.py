"""add ownership, trace context and transactional outbox

Revision ID: 20261012_0005
Revises: 20261009_0004
Create Date: 2026-10-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20261012_0005"
down_revision: str | None = "20261009_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("owner_id", sa.String(length=100), server_default="legacy-admin", nullable=False),
    )
    op.drop_constraint("documents_sha256_key", "documents", type_="unique")
    op.create_unique_constraint("uq_documents_owner_sha256", "documents", ["owner_id", "sha256"])
    op.add_column(
        "chat_sessions",
        sa.Column("owner_id", sa.String(length=100), server_default="legacy-admin", nullable=False),
    )
    op.alter_column("documents", "owner_id", server_default=None)
    op.alter_column("chat_sessions", "owner_id", server_default=None)
    op.add_column("jobs", sa.Column("trace_id", sa.String(length=64), nullable=True))
    op.create_table(
        "outbox_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("aggregate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_outbox_pending",
        "outbox_events",
        ["processed_at", "available_at", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_outbox_pending", table_name="outbox_events")
    op.drop_table("outbox_events")
    op.drop_column("jobs", "trace_id")
    op.drop_column("chat_sessions", "owner_id")
    op.drop_constraint("uq_documents_owner_sha256", "documents", type_="unique")
    op.create_unique_constraint("documents_sha256_key", "documents", ["sha256"])
    op.drop_column("documents", "owner_id")
