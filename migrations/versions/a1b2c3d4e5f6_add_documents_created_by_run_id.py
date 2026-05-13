"""add documents.created_by_run_id FK to processing_runs

Revision ID: a1b2c3d4e5f6
Revises: d8ee42bf7c05
Create Date: 2026-04-28

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "d8ee42bf7c05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("created_by_run_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_documents_created_by_run_id_processing_runs",
        "documents",
        "processing_runs",
        ["created_by_run_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_documents_created_by_run_id",
        "documents",
        ["created_by_run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_documents_created_by_run_id", table_name="documents")
    op.drop_constraint(
        "fk_documents_created_by_run_id_processing_runs",
        "documents",
        type_="foreignkey",
    )
    op.drop_column("documents", "created_by_run_id")
