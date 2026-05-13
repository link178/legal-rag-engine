"""Documents list/detail API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentListItem(BaseModel):
    """One row in GET /v1/documents."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    source_path: str
    title: str | None = None
    checksum: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class DocumentListResponse(BaseModel):
    """GET /v1/documents response."""

    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentListItem]
    count: int
    limit: int
    offset: int


class DocumentDetailResponse(BaseModel):
    """GET /v1/documents/{id} response."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    source_path: str
    title: str | None = None
    checksum: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    chunks_count: int = 0
