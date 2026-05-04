"""Coordinates dense / sparse retrieval and optional RRF fusion."""

from __future__ import annotations

from dataclasses import replace

from app.retrieval.dense import DenseRetriever
from app.retrieval.errors import (
    EmptyQueryError,
    ManifestNotFoundError,
    RetrieverNotConfiguredError,
)
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import RetrievalConfig, RetrievalResultSet, RetrievedChunk
from app.retrieval.sparse import SparseRetriever
from app.storage.postgres.models import IndexManifestRecord


def _finalize_branch(hits: list[RetrievedChunk], *, top_k: int) -> list[RetrievedChunk]:
    """Sort by branch score, take ``top_k``, renumber ``rank_position``; ``rrf_score`` cleared."""
    def sort_key(h: RetrievedChunk) -> tuple:
        score = h.dense_score if h.dense_score is not None else h.sparse_score
        s = score if score is not None else 0.0
        return (-s, h.chunk_id)

    ordered = sorted(hits, key=sort_key)[:top_k]
    out: list[RetrievedChunk] = []
    for i, h in enumerate(ordered, start=1):
        out.append(replace(h, rank_position=i, rrf_score=None))
    return out


class RetrievalOrchestrator:
    """dense_only, sparse_only, or hybrid (RRF) over injected ret."""

    def __init__(
        self,
        dense_retriever: DenseRetriever | None,
        sparse_retriever: SparseRetriever | None,
    ) -> None:
        self._dense = dense_retriever
        self._sparse = sparse_retriever

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig,
        manifest: IndexManifestRecord,
    ) -> RetrievalResultSet:
        if not query.strip():
            raise EmptyQueryError("query must be non-empty")

        mid = manifest.id
        mh = manifest.manifest_hash

        if config.mode == "dense_only":
            if self._dense is None:
                raise RetrieverNotConfiguredError(
                    "dense_only requires a DenseRetriever"
                )
            hits = self._dense.retrieve(query, config, manifest)
            results = _finalize_branch(hits, top_k=config.top_k)
            return RetrievalResultSet(
                query=query,
                mode=config.mode,
                results=results,
                index_manifest_id=mid,
                embedding_provider=manifest.embedding_provider,
                embedding_model=manifest.embedding_model,
                embedding_dimensions=manifest.embedding_dimensions,
                manifest_hash=mh,
                metadata={},
            )

        if config.mode == "sparse_only":
            if self._sparse is None:
                raise RetrieverNotConfiguredError(
                    "sparse_only requires a SparseRetriever"
                )
            hits = self._sparse.retrieve(query, config, manifest)
            results = _finalize_branch(hits, top_k=config.top_k)
            return RetrievalResultSet(
                query=query,
                mode=config.mode,
                results=results,
                index_manifest_id=mid,
                embedding_provider=manifest.embedding_provider,
                embedding_model=manifest.embedding_model,
                embedding_dimensions=manifest.embedding_dimensions,
                manifest_hash=mh,
                metadata={},
            )

        if config.mode == "hybrid":
            if self._dense is None or self._sparse is None:
                raise RetrieverNotConfiguredError(
                    "hybrid requires both DenseRetriever and SparseRetriever"
                )
            if not manifest.include_sparse:
                raise ManifestNotFoundError(
                    "Hybrid retrieval needs a manifest built with sparse indexing "
                    "(include_sparse=true). Re-run: python -m app.indexing.cli without --no-sparse"
                )
            d_hits = self._dense.retrieve(query, config, manifest)
            s_hits = self._sparse.retrieve(query, config, manifest)
            lists = [d_hits, s_hits]
            non_empty = [x for x in lists if x]
            fused = reciprocal_rank_fusion(
                non_empty,
                rrf_k=config.rrf_k,
                top_k=config.top_k,
            )
            return RetrievalResultSet(
                query=query,
                mode=config.mode,
                results=fused,
                index_manifest_id=mid,
                embedding_provider=manifest.embedding_provider,
                embedding_model=manifest.embedding_model,
                embedding_dimensions=manifest.embedding_dimensions,
                manifest_hash=mh,
                metadata={},
            )

        raise ValueError(f"unsupported mode {config.mode!r}")
