"""Index / manifest API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class IndexRequest(BaseModel):
    """POST /v1/index body (optional overrides; remainder from Settings — CLI parity)."""

    model_config = ConfigDict(extra="forbid")

    chunking_strategy: str | None = None
    include_dense: bool = True
    include_sparse: bool = True
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = Field(default=None, gt=0)
    batch_size: int = Field(default=32, gt=0, le=512)
    force_reindex: bool = False
    corpus_version: str | None = None

    @model_validator(mode="after")
    def at_least_dense_or_sparse(self) -> IndexRequest:
        if not self.include_dense and not self.include_sparse:
            raise ValueError("at least one of include_dense or include_sparse must be True")
        return self


class IndexResponse(BaseModel):
    """POST /v1/index response."""

    model_config = ConfigDict(extra="forbid")

    manifest_id: UUID | None = None
    manifest_hash: str | None = None
    config_hash: str | None = None
    chunk_set_hash: str | None = None
    chunking_strategy: str | None = None
    embedding_provider: str
    embedding_model: str | None = None
    embedding_dimensions: int
    include_dense: bool
    include_sparse: bool
    embeddings_persisted: bool
    chunk_count: int = 0
    indexed_chunk_count: int = 0
    failed_chunks_count: int = 0
    processing_run_id: UUID | None = None
    created: bool
    skipped_existing: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexManifestItemResponse(BaseModel):
    """One row in GET /v1/index-manifests (mirrors index_manifests table)."""

    model_config = ConfigDict(extra="forbid")

    manifest_id: UUID
    manifest_hash: str
    config_hash: str
    chunk_set_hash: str
    chunking_strategy: str | None = None
    embedding_provider: str
    embedding_model: str | None = None
    embedding_dimensions: int
    include_dense: bool
    include_sparse: bool
    embeddings_persisted: bool
    document_count: int
    chunk_count: int
    indexed_chunk_count: int
    corpus_version: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexManifestDetailResponse(IndexManifestItemResponse):
    """GET /v1/index-manifests/{manifest_id}."""

    pass


class IndexManifestListResponse(BaseModel):
    """GET /v1/index-manifests response."""

    model_config = ConfigDict(extra="forbid")

    manifests: list[IndexManifestItemResponse]
    count: int
    limit: int
    offset: int
