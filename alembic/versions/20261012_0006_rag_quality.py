"""add document metadata, structured chunks and lexical search index

Revision ID: 20261012_0006
Revises: 20261012_0005
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20261012_0006"
down_revision: str | None = "20261012_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("department", sa.String(length=100), nullable=True))
    op.add_column("documents", sa.Column("document_type", sa.String(length=100), nullable=True))
    op.add_column("documents", sa.Column("document_date", sa.Date(), nullable=True))
    op.add_column(
        "documents", sa.Column("access_level", sa.Integer(), server_default="0", nullable=False)
    )
    op.add_column(
        "documents",
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.create_check_constraint("ck_documents_access_level", "documents", "access_level >= 0")
    op.create_index(
        "ix_documents_metadata",
        "documents",
        ["department", "document_type", "document_date"],
    )
    op.alter_column("documents", "access_level", server_default=None)
    op.alter_column("documents", "tags", server_default=None)

    op.add_column("document_chunks", sa.Column("section", sa.String(length=255), nullable=True))
    op.add_column(
        "document_chunks",
        sa.Column("content_type", sa.String(length=32), server_default="text", nullable=False),
    )
    op.alter_column("document_chunks", "content_type", server_default=None)
    op.execute(
        "CREATE INDEX ix_document_chunks_fts ON document_chunks "
        "USING gin (to_tsvector('simple', text))"
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_fts", table_name="document_chunks")
    op.drop_column("document_chunks", "content_type")
    op.drop_column("document_chunks", "section")
    op.drop_index("ix_documents_metadata", table_name="documents")
    op.drop_constraint("ck_documents_access_level", "documents", type_="check")
    op.drop_column("documents", "tags")
    op.drop_column("documents", "access_level")
    op.drop_column("documents", "document_date")
    op.drop_column("documents", "document_type")
    op.drop_column("documents", "department")
