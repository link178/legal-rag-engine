"""Ingest API schemas."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class IngestRequest(BaseModel):
    """POST /v1/ingest body."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1)
    persist: bool = True


class IngestResponse(BaseModel):
    """POST /v1/ingest response."""

    model_config = ConfigDict(extra="forbid")

    persisted: bool
    document_id: UUID | None = None
    processing_run_id: UUID | None = None
    source_path: str
    source_type: str = ""
    checksum: str = ""
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created: bool | None = None
    skipped: bool | None = None
    error: str | None = None
