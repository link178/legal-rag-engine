"""ContextBuilder from RetrievedChunk."""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from app.generation.context import ContextBuilder
from app.retrieval.models import RetrievedChunk


def _hit(
    text: str,
    *,
    rank: int = 1,
    dense: float | None = 0.5,
    sparse: float | None = None,
    rrf: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        text=text,
        source_path="a.md",
        title=None,
        heading=None,
        chunk_index=0,
        chunking_strategy="fixed_size",
        dense_score=dense,
        sparse_score=sparse,
        rrf_score=rrf,
        rank_position=rank,
        retrieval_sources=("dense",),
    )


def test_citation_ids_start_at_one_and_preserve_order() -> None:
    h1 = _hit("one", rank=1)
    h2 = _hit("two", rank=2)
    b = ContextBuilder(max_chunks=10, max_context_chars=10000, max_chunk_chars=1000)
    out = b.build([h1, h2])
    assert [x.citation_id for x in out] == [1, 2]
    assert [x.text for x in out] == ["one", "two"]
    assert out[0].chunk_id == h1.chunk_id


def test_respects_max_chunks() -> None:
    hits = [_hit(str(i), rank=i + 1) for i in range(10)]
    b = ContextBuilder(max_chunks=3, max_context_chars=10000, max_chunk_chars=100)
    assert len(b.build(hits)) == 3


def test_respects_max_context_chars() -> None:
    hits = [_hit("x" * 100, rank=i + 1) for i in range(20)]
    b = ContextBuilder(max_chunks=20, max_context_chars=250, max_chunk_chars=200)
    out = b.build(hits)
    total = sum(len(x.text) for x in out)
    assert total <= 250
    assert len(out) >= 1


def test_truncates_long_chunks_with_ellipsis() -> None:
    long = "y" * 50
    h = _hit(long, rank=1)
    b = ContextBuilder(max_chunks=5, max_context_chars=10000, max_chunk_chars=10)
    out = b.build([h])
    assert len(out) == 1
    assert out[0].text.endswith("\u2026")
    assert len(out[0].text) == 10


def test_min_score_filters() -> None:
    low = _hit("a", rank=1, dense=0.1)
    high = _hit("b", rank=2, dense=0.9)
    b = ContextBuilder(min_score=0.5, max_chunks=10, max_context_chars=10000, max_chunk_chars=1000)
    out = b.build([low, high])
    assert len(out) == 1
    assert out[0].text == "b"


def test_min_score_skips_none_score() -> None:
    h = _hit("z", rank=1, dense=None, sparse=None, rrf=None)
    b = ContextBuilder(min_score=0.1, max_chunks=5, max_context_chars=1000, max_chunk_chars=100)
    assert b.build([h]) == []


def test_no_mutation_of_input_chunks() -> None:
    h = _hit("original" * 20, rank=1)
    before = deepcopy(h)
    b = ContextBuilder(max_chunks=1, max_context_chars=10000, max_chunk_chars=5)
    _ = b.build([h])
    assert h.text == before.text
    assert h.chunk_id == before.chunk_id


def test_rrf_priority_for_score_when_present() -> None:
    h = _hit("x", rank=1, dense=0.99, sparse=0.01, rrf=0.02)
    b = ContextBuilder(max_chunks=1, max_context_chars=50, max_chunk_chars=40)
    out = b.build([h])
    assert out[0].score == 0.02
