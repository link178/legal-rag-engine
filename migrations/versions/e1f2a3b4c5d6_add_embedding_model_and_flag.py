"""add index_manifests embedding_model and embeddings_persisted

Revision ID: e1f2a3b4c5d6
Revises: c3d4e5f6a7b8
Create Date: 2026-04-30

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: str | Sequence[str] | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "index_manifests",
        sa.Column("embedding_model", sa.Text(), nullable=True),
    )
    op.add_column(
        "index_manifests",
        sa.Column(
            "embeddings_persisted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("index_manifests", "embeddings_persisted")
    op.drop_column("index_manifests", "embedding_model")
