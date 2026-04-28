"""SQLAlchemy ORM models for Postgres."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.storage.postgres.base import Base


class ProcessingRunRecord(Base):
    """Operator-facing run trace (not a distributed job queue)."""

    __tablename__ = "processing_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    documents_processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("ix_processing_runs_run_type", "run_type"),
        Index("ix_processing_runs_status", "status"),
    )


class DocumentRecord(Base):
    """Persisted document row."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    checksum: Mapped[str] = mapped_column(Text, nullable=False)
    created_by_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("processing_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    chunks: Mapped[list[ChunkRecord]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_documents_checksum", "checksum"),
        Index("ix_documents_source_path", "source_path"),
        Index("ix_documents_created_by_run_id", "created_by_run_id"),
        UniqueConstraint(
            "source_path",
            "checksum",
            name="uq_documents_source_path_checksum",
        ),
    )


class ChunkRecord(Base):
    """Persisted chunk row."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_by_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("processing_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    # Python name ``chunk_text``; DB column remains ``text`` (``text`` shadows ``sqlalchemy.text``).
    chunk_text: Mapped[str] = mapped_column("text", Text(), nullable=False)
    heading: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunking_strategy: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    token_estimate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    checksum: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    document: Mapped[DocumentRecord] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_created_by_run_id", "created_by_run_id"),
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_chunking_strategy", "chunking_strategy"),
        UniqueConstraint(
            "document_id",
            "chunk_index",
            "chunking_strategy",
            name="uq_chunks_document_chunk_strategy",
        ),
    )


class IndexManifestRecord(Base):
    """One indexing build snapshot (Phase 4A: traceability only; no pgvector column here)."""

    __tablename__ = "index_manifests"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("processing_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    corpus_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunking_strategy: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_provider: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    document_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    indexed_chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    include_sparse: Mapped[bool] = mapped_column(Boolean, nullable=False)
    include_dense: Mapped[bool] = mapped_column(Boolean, nullable=False)
    config_hash: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_set_hash: Mapped[str] = mapped_column(Text, nullable=False)
    manifest_hash: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("ix_index_manifests_run_id", "run_id"),
        Index("ix_index_manifests_config_hash", "config_hash"),
        Index("ix_index_manifests_chunk_set_hash", "chunk_set_hash"),
        Index("ix_index_manifests_manifest_hash", "manifest_hash"),
    )


class IndexManifestChunkRecord(Base):
    """Per-chunk row for one manifest (status + optional sparse JSON)."""

    __tablename__ = "index_manifest_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    manifest_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("index_manifests.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    dense_indexed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    sparse_indexed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    sparse_terms_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("ix_index_manifest_chunks_manifest_id", "manifest_id"),
        Index("ix_index_manifest_chunks_chunk_id", "chunk_id"),
        UniqueConstraint(
            "manifest_id",
            "chunk_id",
            name="uq_index_manifest_chunks_manifest_chunk",
        ),
    )
