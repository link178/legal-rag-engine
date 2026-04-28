"""add chunks.created_by_run_id FK to processing_runs

Revision ID: f7e8d9c0b1a2
Revises: a1b2c3d4e5f6
Create Date: 2026-04-28

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f7e8d9c0b1a2"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("created_by_run_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_chunks_created_by_run_id_processing_runs",
        "chunks",
        "processing_runs",
        ["created_by_run_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_chunks_created_by_run_id",
        "chunks",
        ["created_by_run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_created_by_run_id", table_name="chunks")
    op.drop_constraint(
        "fk_chunks_created_by_run_id_processing_runs",
        "chunks",
        type_="foreignkey",
    )
    op.drop_column("chunks", "created_by_run_id")
