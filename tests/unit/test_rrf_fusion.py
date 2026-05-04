"""Reciprocal Rank Fusion (pure) tests."""

from __future__ import annotations

from uuid import uuid4

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import RetrievedChunk


def _ch(
    i: int,
    *,
    dense: float | None = None,
    sparse: float | None = None,
    src: tuple[str, ...] = (),
) -> RetrievedChunk:
    u = uuid4()
    return RetrievedChunk(
        chunk_id=u,
        document_id=uuid4(),
        text=f"c{i}",
        source_path=None,
        title=None,
        heading=None,
        chunk_index=i,
        chunking_strategy="fixed_size",
        dense_score=dense,
        sparse_score=sparse,
        rank_position=i,
        retrieval_sources=src,
    )


def test_single_list_order_matches_rrf_formula() -> None:
    a, b = _ch(0, dense=0.9, src=("dense",)), _ch(1, dense=0.8, src=("dense",))
    ranked = [a, b]
    out = reciprocal_rank_fusion([ranked], rrf_k=60, top_k=10)
    assert [x.chunk_id for x in out] == [a.chunk_id, b.chunk_id]
    assert out[0].rrf_score > out[1].rrf_score


def test_chunk_in_both_lists_ranks_higher() -> None:
    c_shared = _ch(0, dense=0.5, sparse=0.3)
    c_dense_only = _ch(1, dense=0.9)
    c_sparse_only = _ch(2, sparse=0.9)
    dense_list = [c_shared, c_dense_only]
    sparse_list = [c_sparse_only, c_shared]
    orig_ids = {c_shared.chunk_id, c_dense_only.chunk_id, c_sparse_only.chunk_id}
    out = reciprocal_rank_fusion([dense_list, sparse_list], rrf_k=60, top_k=10)
    assert out[0].chunk_id == c_shared.chunk_id
    assert {x.chunk_id for x in out} == orig_ids
    shared_hit = next(x for x in out if x.chunk_id == c_shared.chunk_id)
    assert shared_hit.dense_score == 0.5
    assert shared_hit.sparse_score == 0.3
    assert set(shared_hit.retrieval_sources) == {"dense", "sparse"}


def test_top_k_truncates() -> None:
    chunks = [_ch(i, dense=1.0 / (i + 1)) for i in range(5)]
    out = reciprocal_rank_fusion([chunks], rrf_k=60, top_k=2)
    assert len(out) == 2


def test_inputs_not_mutated() -> None:
    a = _ch(0, dense=0.9, src=("dense",))
    b = _ch(1, dense=0.8, src=("dense",))
    ranked = [a, b]
    before_a_rrf = a.rrf_score
    reciprocal_rank_fusion([ranked], rrf_k=60, top_k=10)
    assert a.rrf_score == before_a_rrf
    assert ranked[0] is a
