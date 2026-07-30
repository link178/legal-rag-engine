"""Reciprocal Rank Fusion (pure)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from uuid import UUID

from app.retrieval.models import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    rrf_k: int,
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Fuse ranked lists by RRF: score += 1 / (rrf_k + rank) per list (rank 1-based).

    When two non-empty lists are passed, the first is dense and the second is sparse.
    When one list is passed, each chunk should set ``retrieval_sources`` to ("dense",)
    or ("sparse",), or set only the corresponding branch score.
    """
    lists = [list(seq) for seq in ranked_lists if seq]
    if not lists:
        return []

    if len(lists) == 2:
        label_seq: tuple[str, ...] = ("dense", "sparse")
    else:
        label_seq = _infer_single_list_labels(lists[0])

    rrf_accum: dict[UUID, float] = defaultdict(float)
    sources: dict[UUID, set[str]] = defaultdict(set)
    merged: dict[UUID, RetrievedChunk] = {}

    for label, ranked in zip(label_seq, lists, strict=True):
        for rank, ch in enumerate(ranked, start=1):
            cid = ch.chunk_id
            rrf_accum[cid] += 1.0 / (rrf_k + rank)
            sources[cid].add(label)
            if cid not in merged:
                merged[cid] = ch
            else:
                merged[cid] = _merge_hit(merged[cid], ch)

    ordered = sorted(
        merged.keys(),
        key=lambda u: (
            -rrf_accum[u],
            merged[u].source_path or "",
            merged[u].chunk_index if merged[u].chunk_index is not None else -1,
            str(u),
        ),
    )[:top_k]

    out: list[RetrievedChunk] = []
    for pos, cid in enumerate(ordered, start=1):
        src_tuple = tuple(sorted(sources[cid]))
        base = merged[cid]
        out.append(
            replace(
                base,
                rrf_score=rrf_accum[cid],
                rank_position=pos,
                retrieval_sources=src_tuple,
            )
        )
    return out


def _infer_single_list_labels(ranked: Sequence[RetrievedChunk]) -> tuple[str, ...]:
    if not ranked:
        return ("dense",)
    ch0 = ranked[0]
    if ch0.retrieval_sources:
        return tuple(ch0.retrieval_sources)
    if ch0.dense_score is not None and ch0.sparse_score is None:
        return ("dense",)
    if ch0.sparse_score is not None and ch0.dense_score is None:
        return ("sparse",)
    return ("dense",)


def _merge_hit(a: RetrievedChunk, b: RetrievedChunk) -> RetrievedChunk:
    d = a.dense_score if a.dense_score is not None else b.dense_score
    s = a.sparse_score if a.sparse_score is not None else b.sparse_score
    meta = dict(a.metadata)
    for k, v in b.metadata.items():
        if k not in meta:
            meta[k] = v
    return replace(a, dense_score=d, sparse_score=s, metadata=meta)
