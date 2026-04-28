"""add index_manifests and index_manifest_chunks

Revision ID: c3d4e5f6a7b8
Revises: f7e8d9c0b1a2
Create Date: 2026-04-28

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c3d4e5f6a7b8"
down_revision: str | Sequence[str] | None = "f7e8d9c0b1a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "index_manifests",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("corpus_version", sa.Text(), nullable=True),
        sa.Column("chunking_strategy", sa.Text(), nullable=True),
        sa.Column("embedding_provider", sa.Text(), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("document_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("indexed_chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("include_sparse", sa.Boolean(), nullable=False),
        sa.Column("include_dense", sa.Boolean(), nullable=False),
        sa.Column("config_hash", sa.Text(), nullable=False),
        sa.Column("chunk_set_hash", sa.Text(), nullable=False),
        sa.Column("manifest_hash", sa.Text(), nullable=False),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["processing_runs.id"],
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_index_manifests_run_id", "index_manifests", ["run_id"])
    op.create_index("ix_index_manifests_config_hash", "index_manifests", ["config_hash"])
    op.create_index("ix_index_manifests_chunk_set_hash", "index_manifests", ["chunk_set_hash"])
    op.create_index("ix_index_manifests_manifest_hash", "index_manifests", ["manifest_hash"])

    op.create_table(
        "index_manifest_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("manifest_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dense_indexed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("sparse_indexed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "sparse_terms_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["chunks.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["manifest_id"],
            ["index_manifests.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "manifest_id",
            "chunk_id",
            name="uq_index_manifest_chunks_manifest_chunk",
        ),
    )
    op.create_index(
        "ix_index_manifest_chunks_manifest_id",
        "index_manifest_chunks",
        ["manifest_id"],
    )
    op.create_index(
        "ix_index_manifest_chunks_chunk_id",
        "index_manifest_chunks",
        ["chunk_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_index_manifest_chunks_chunk_id", table_name="index_manifest_chunks")
    op.drop_index("ix_index_manifest_chunks_manifest_id", table_name="index_manifest_chunks")
    op.drop_table("index_manifest_chunks")

    op.drop_index("ix_index_manifests_manifest_hash", table_name="index_manifests")
    op.drop_index("ix_index_manifests_chunk_set_hash", table_name="index_manifests")
    op.drop_index("ix_index_manifests_config_hash", table_name="index_manifests")
    op.drop_index("ix_index_manifests_run_id", table_name="index_manifests")
    op.drop_table("index_manifests")
