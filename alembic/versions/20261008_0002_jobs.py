"""add background jobs

Revision ID: 20261008_0002
Revises: 20261008_0001
Create Date: 2026-10-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20261008_0002"
down_revision: str | None = "20261008_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "type",
            sa.Enum(
                "INGESTION",
                name="job_type",
                native_enum=False,
                create_constraint=True,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "PARSING",
                "CHUNKING",
                "EMBEDDING",
                "INDEXING",
                "READY",
                "FAILED",
                name="job_status",
                native_enum=False,
                create_constraint=True,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("attempt >= 0", name="ck_jobs_attempt"),
        sa.CheckConstraint("progress BETWEEN 0 AND 100", name="ck_jobs_progress"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_jobs_document_id", "jobs", ["document_id"], unique=False)
    op.create_index("ix_jobs_status_updated_at", "jobs", ["status", "updated_at"], unique=False)

    op.execute(
        sa.text(
            """
            INSERT INTO jobs (
                id, document_id, type, status, progress, attempt,
                error_code, error_message, created_at, started_at, finished_at, updated_at
            )
            SELECT
                gen_random_uuid(),
                id,
                'INGESTION',
                status,
                CASE
                    WHEN status IN ('READY', 'FAILED') THEN 100
                    WHEN status = 'INDEXING' THEN 80
                    WHEN status = 'EMBEDDING' THEN 50
                    WHEN status = 'CHUNKING' THEN 30
                    WHEN status = 'PARSING' THEN 10
                    ELSE 0
                END,
                0,
                NULL,
                NULL,
                created_at,
                CASE WHEN status = 'PENDING' THEN NULL ELSE created_at END,
                CASE WHEN status IN ('READY', 'FAILED') THEN updated_at ELSE NULL END,
                updated_at
            FROM documents
            """
        )
    )


def downgrade() -> None:
    op.drop_index("ix_jobs_status_updated_at", table_name="jobs")
    op.drop_index("ix_jobs_document_id", table_name="jobs")
    op.drop_table("jobs")
