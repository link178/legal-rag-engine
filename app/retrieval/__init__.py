"""Hybrid retrieval: dense, sparse, RRF fusion, orchestration (Phase 5)."""

from __future__ import annotations

from app.retrieval.dense import DenseRetriever
from app.retrieval.errors import (
    EmbeddingDimensionMismatchError,
    EmptyQueryError,
    ManifestNotFoundError,
    RetrievalError,
    RetrieverNotConfiguredError,
)
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.manifest import resolve_manifest_record
from app.retrieval.models import RetrievalConfig, RetrievalResultSet, RetrievedChunk
from app.retrieval.orchestrator import RetrievalOrchestrator
from app.retrieval.sparse import SparseRetriever

__all__ = [
    "DenseRetriever",
    "EmbeddingDimensionMismatchError",
    "EmptyQueryError",
    "ManifestNotFoundError",
    "RetrievalError",
    "RetrieverNotConfiguredError",
    "RetrievalConfig",
    "RetrievalOrchestrator",
    "RetrievalResultSet",
    "RetrievedChunk",
    "SparseRetriever",
    "reciprocal_rank_fusion",
    "resolve_manifest_record",
]
