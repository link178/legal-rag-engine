"""Hybrid retrieval: dense, sparse, RRF fusion, orchestration (Phase 5)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.retrieval.errors import (
    EmbeddingDimensionMismatchError,
    EmptyQueryError,
    ManifestNotFoundError,
    RetrievalError,
    RetrieverNotConfiguredError,
)
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import (
    RetrievalConfig,
    RetrievalMetadataFilter,
    RetrievalResultSet,
    RetrievedChunk,
)

if TYPE_CHECKING:
    from app.retrieval.dense import DenseRetriever
    from app.retrieval.manifest import resolve_manifest_record
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
    "RetrievalMetadataFilter",
    "RetrievalOrchestrator",
    "RetrievalResultSet",
    "RetrievedChunk",
    "SparseRetriever",
    "reciprocal_rank_fusion",
    "resolve_manifest_record",
]


def __getattr__(name: str):
    if name == "DenseRetriever":
        from app.retrieval.dense import DenseRetriever

        return DenseRetriever
    if name == "SparseRetriever":
        from app.retrieval.sparse import SparseRetriever

        return SparseRetriever
    if name == "RetrievalOrchestrator":
        from app.retrieval.orchestrator import RetrievalOrchestrator

        return RetrievalOrchestrator
    if name == "resolve_manifest_record":
        from app.retrieval.manifest import resolve_manifest_record

        return resolve_manifest_record
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
