"""Indexing configuration, manifest, and hashing (pure domain types; no SQLAlchemy)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.domain.models import ProcessingRun


@dataclass(frozen=True, slots=True)
class IndexingConfig:
    """Validated parameters for a single indexing job (Phase 4A foundation)."""

    embedding_provider: str = "deterministic_hash"
    embedding_dimensions: int = 16
    # Model id for providers that need one (e.g. local_sentence_transformers); None/blank if N/A.
    embedding_model: str | None = None
    chunking_strategy: str | None = None  # filter; None means all persisted strategies
    batch_size: int = 32
    include_sparse: bool = True
    include_dense: bool = True
    force_reindex: bool = False

    def __post_init__(self) -> None:
        if not self.embedding_provider.strip():
            raise ValueError("embedding_provider must be non-empty")
        if self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not self.include_sparse and not self.include_dense:
            raise ValueError("at least one of include_sparse or include_dense must be True")

    def normalized_embedding_model(self) -> str | None:
        if self.embedding_model is None:
            return None
        s = self.embedding_model.strip()
        return s or None

    def normalized_dict(self) -> dict[str, Any]:
        """Canonical dict for config_hash (sorted keys in JSON)."""
        return {
            "batch_size": self.batch_size,
            "chunking_strategy": self.chunking_strategy.strip()
            if self.chunking_strategy
            else None,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_model": self.normalized_embedding_model(),
            "embedding_provider": self.embedding_provider.strip(),
            "include_dense": self.include_dense,
            "include_sparse": self.include_sparse,
        }

    def config_hash(self) -> str:
        """SHA-256 hex of canonical JSON (sorted keys). Excludes force_reindex."""
        payload = json.dumps(self.normalized_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_chunk_row(
    chunk_id: UUID,
    *,
    chunking_strategy: str,
    checksum: str | None,
) -> str:
    """Stable string for one chunk row used in chunk_set_hash."""
    parts = (
        str(chunk_id),
        chunking_strategy.strip(),
        checksum or "",
    )
    return "|".join(parts)


def chunk_set_hash_from_fingerprints(fingerprints: list[str]) -> str:
    """Deterministic hash over the selected chunk set (order-independent)."""
    unique_sorted = sorted(set(fingerprints))
    payload = json.dumps(unique_sorted, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_hash(config_hash: str, chunk_set_hash: str) -> str:
    """Combines config and chunk set for idempotent skip."""
    payload = f"{config_hash}:{chunk_set_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class IndexManifest:
    """In-memory representation of a persisted index manifest (mirrors DB row)."""

    embedding_provider: str
    embedding_dimensions: int
    include_sparse: bool
    include_dense: bool
    config_hash: str
    chunk_set_hash: str
    manifest_hash: str
    id: UUID | None = None
    run_id: UUID | None = None
    corpus_version: str | None = None
    chunking_strategy: str | None = None
    document_count: int = 0
    chunk_count: int = 0
    indexed_chunk_count: int = 0
    embedding_model: str | None = None
    embeddings_persisted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: Any = None  # datetime when loaded from DB
    updated_at: Any = None


@dataclass(slots=True)
class IndexedChunkResult:
    """Per-chunk outcome from in-memory indexing (dense/sparse flags, optional error)."""

    chunk_id: UUID
    dense_indexed: bool
    sparse_indexed: bool
    error: str | None = None


@dataclass(slots=True)
class IndexingRunResult:
    """Outcome of ``index_chunks_persisted``."""

    manifest: IndexManifest | None
    run: ProcessingRun | None = None
    skipped_existing: bool = False
    chunk_results: list[IndexedChunkResult] = field(default_factory=list)
    error: str | None = None
