"""Pure domain entities for documents, chunks, and processing runs.

No FastAPI, SQLAlchemy, or Alembic imports here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


def _empty_metadata() -> dict[str, Any]:
    return {}


@dataclass(slots=True)
class Document:
    """Logical document prior to or after persistence."""

    source_path: str
    source_type: str
    checksum: str
    id: UUID | None = None
    title: str | None = None
    raw_text: str | None = None
    normalized_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)
    created_by_run_id: UUID | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.source_path.strip():
            raise ValueError("source_path must be non-empty")
        if not self.source_type.strip():
            raise ValueError("source_type must be non-empty")
        if not self.checksum.strip():
            raise ValueError("checksum must be non-empty")


@dataclass(slots=True)
class Chunk:
    """Indexable fragment belonging to a document."""

    document_id: UUID
    chunk_index: int
    text: str
    chunking_strategy: str
    char_count: int
    id: UUID | None = None
    created_by_run_id: UUID | None = None
    heading: str | None = None
    page_number: int | None = None
    token_estimate: int | None = None
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)
    checksum: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be >= 0")
        if self.char_count < 0:
            raise ValueError("char_count must be >= 0")
        if not self.text.strip():
            raise ValueError("text must be non-empty")
        if not self.chunking_strategy.strip():
            raise ValueError("chunking_strategy must be non-empty")


@dataclass(slots=True)
class ProcessingRun:
    """Trace record for operator-triggered pipeline steps (not a job scheduler)."""

    run_type: str
    status: str
    id: UUID | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    documents_processed: int = 0
    chunks_created: int = 0
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.run_type.strip():
            raise ValueError("run_type must be non-empty")
        if not self.status.strip():
            raise ValueError("status must be non-empty")
        if self.documents_processed < 0:
            raise ValueError("documents_processed must be >= 0")
        if self.chunks_created < 0:
            raise ValueError("chunks_created must be >= 0")
