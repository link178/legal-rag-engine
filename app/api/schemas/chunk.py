"""Chunk API schemas."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkRequest(BaseModel):
    """POST /v1/chunk body."""

    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    strategy: Literal["fixed_size", "structure_aware"] = "fixed_size"
    chunk_size: int = Field(default=1200, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    min_chunk_chars: int = Field(default=80, ge=0)
    preserve_headings: bool = True
    include_chunks: bool = False
    chunk_preview_chars: int = Field(default=240, gt=0, le=2000)

    @model_validator(mode="after")
    def overlap_below_size(self) -> ChunkRequest:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        return self


class ChunkItemResponse(BaseModel):
    """One chunk row exposed by the API."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    chunking_strategy: str
    char_count: int
    heading: str | None = None
    page_number: int | None = None
    checksum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    text_preview: str


class ChunkResponse(BaseModel):
    """POST /v1/chunk response."""

    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    processing_run_id: UUID | None = None
    strategy: str
    config_hash: str
    created: bool
    skipped_existing: bool
    chunks_count: int
    chunk_ids: list[UUID] = Field(default_factory=list)
    chunks: list[ChunkItemResponse] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunksResponse(BaseModel):
    """GET /v1/documents/{document_id}/chunks response."""

    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    strategy: str | None = None
    count: int
    limit: int
    offset: int
    chunks: list[ChunkItemResponse] = Field(default_factory=list)
