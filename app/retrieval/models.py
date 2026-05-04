"""Retrieval configuration and result types (pure; no FastAPI / SQLAlchemy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.core.constants import SUPPORTED_RETRIEVAL_MODES


def _empty_metadata() -> dict[str, Any]:
    return {}


@dataclass(frozen=True, kw_only=True, slots=True)
class RetrievalConfig:
    """Parameters for one retrieval request."""

    mode: str = "hybrid"
    top_k: int = 5
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rrf_k: int = 60
    index_manifest_id: UUID | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    chunking_strategy: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_RETRIEVAL_MODES:
            raise ValueError(
                f"mode must be one of {SUPPORTED_RETRIEVAL_MODES}, got {self.mode!r}"
            )
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.dense_top_k <= 0:
            raise ValueError("dense_top_k must be > 0")
        if self.sparse_top_k <= 0:
            raise ValueError("sparse_top_k must be > 0")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be > 0")
        if self.embedding_dimensions is not None and self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be > 0 when set")


@dataclass(frozen=True, kw_only=True, slots=True)
class RetrievedChunk:
    """One ranked hit after dense and/or sparse retrieval (before or after RRF)."""

    chunk_id: UUID
    document_id: UUID
    text: str
    source_path: str | None
    title: str | None
    heading: str | None
    chunk_index: int
    chunking_strategy: str
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rank_position: int = 0
    retrieval_sources: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be >= 0")
        if not self.chunking_strategy.strip():
            raise ValueError("chunking_strategy must be non-empty")


@dataclass(frozen=True, kw_only=True, slots=True)
class RetrievalResultSet:
    """Full retrieval output for a query."""

    query: str
    mode: str
    results: list[RetrievedChunk]
    index_manifest_id: UUID | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    manifest_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)
