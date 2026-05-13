"""SparseRetriever BM25-lite scoring (mocked rows)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from app.retrieval.models import RetrievalConfig, RetrievalMetadataFilter
from app.retrieval.sparse import SparseRetriever


def _row(text: str, terms: dict[str, int]) -> tuple[MagicMock, MagicMock, dict[str, int]]:
    cid = uuid4()
    did = uuid4()
    ch = MagicMock()
    ch.id = cid
    ch.document_id = did
    ch.chunk_text = text
    ch.heading = None
    ch.chunk_index = 0
    ch.chunking_strategy = "fixed_size"
    doc = MagicMock()
    doc.source_path = "x.md"
    doc.title = None
    return ch, doc, dict(terms)


@pytest.fixture
def manifest():
    m = MagicMock()
    m.id = uuid4()
    m.manifest_hash = "h" * 64
    return m


def test_overlap_positive_score(manifest) -> None:
    rows = [
        _row("cat dog", {"cat": 1, "dog": 1}),
        _row("bird", {"bird": 1}),
    ]
    with patch("app.retrieval.sparse.IndexManifestRepository") as MockRepo:
        MockRepo.return_value.list_sparse_chunk_rows.return_value = rows
        ret = SparseRetriever(MagicMock())
        cfg = RetrievalConfig(mode="sparse_only", sparse_top_k=10)
        out = ret.retrieve("cat", cfg, manifest)
        inst = MockRepo.return_value
        inst.list_sparse_chunk_rows.assert_called_once()
        assert inst.list_sparse_chunk_rows.call_args.kwargs.get("metadata_filter") is None
    assert len(out) >= 1
    assert out[0].sparse_score is not None
    assert out[0].sparse_score > 0


def test_no_overlap_zero(manifest) -> None:
    rows = [_row("zzz", {"zzz": 1})]
    with patch("app.retrieval.sparse.IndexManifestRepository") as MockRepo:
        MockRepo.return_value.list_sparse_chunk_rows.return_value = rows
        ret = SparseRetriever(MagicMock())
        out = ret.retrieve("nomatchhere", RetrievalConfig(), manifest)
    assert len(out) == 1
    assert out[0].sparse_score == 0.0


def test_case_insensitive(manifest) -> None:
    rows = [_row("Hello", {"hello": 2})]
    with patch("app.retrieval.sparse.IndexManifestRepository") as MockRepo:
        MockRepo.return_value.list_sparse_chunk_rows.return_value = rows
        ret = SparseRetriever(MagicMock())
        a = ret.retrieve("HELLO", RetrievalConfig(), manifest)[0].sparse_score
        b = ret.retrieve("hello", RetrievalConfig(), manifest)[0].sparse_score
    assert a == b


def test_repeated_query_terms_adds_score(manifest) -> None:
    rows = [_row("x", {"cat": 1})]
    with patch("app.retrieval.sparse.IndexManifestRepository") as MockRepo:
        MockRepo.return_value.list_sparse_chunk_rows.return_value = rows
        ret = SparseRetriever(MagicMock())
        one = ret.retrieve("cat", RetrievalConfig(), manifest)[0].sparse_score
        two = ret.retrieve("cat cat", RetrievalConfig(), manifest)[0].sparse_score
    assert two > one


def test_sparse_top_k(manifest) -> None:
    rows = [_row(f"c{i}", {f"t{i}": 1}) for i in range(5)]
    with patch("app.retrieval.sparse.IndexManifestRepository") as MockRepo:
        MockRepo.return_value.list_sparse_chunk_rows.return_value = rows
        ret = SparseRetriever(MagicMock())
        q = " ".join(f"t{i}" for i in range(5))
        out = ret.retrieve(q, RetrievalConfig(sparse_top_k=2), manifest)
    assert len(out) == 2


def test_sparse_passes_metadata_filter(manifest) -> None:
    rows = [_row("c0", {"t0": 1})]
    mf = RetrievalMetadataFilter(jurisdiction="es")
    with patch("app.retrieval.sparse.IndexManifestRepository") as MockRepo:
        MockRepo.return_value.list_sparse_chunk_rows.return_value = rows
        ret = SparseRetriever(MagicMock())
        ret.retrieve("t0", RetrievalConfig(metadata_filter=mf), manifest)
    kw = MockRepo.return_value.list_sparse_chunk_rows.call_args
    assert kw[0][0] == manifest.id
    assert kw.kwargs.get("metadata_filter") is mf

