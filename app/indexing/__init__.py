"""Indexing: dense/sparse foundations, service, and sync placeholder."""

from __future__ import annotations

from app.indexing.models import (
    IndexedChunkResult,
    IndexingConfig,
    IndexingRunResult,
    IndexManifest,
)
from app.indexing.runner import index_chunks_persisted
from app.indexing.service import IndexingService, default_indexing_service
from app.indexing.sync import noop_sync_placeholder

__all__ = [
    "IndexManifest",
    "IndexedChunkResult",
    "IndexingConfig",
    "IndexingRunResult",
    "IndexingService",
    "default_indexing_service",
    "index_chunks_persisted",
    "noop_sync_placeholder",
]
