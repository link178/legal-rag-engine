"""DenseRetriever with mocked vector search."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from app.retrieval.dense import DenseRetriever
from app.retrieval.errors import EmptyQueryError
from app.retrieval.models import RetrievalConfig


@pytest.fixture
def manifest():
    m = MagicMock()
    m.id = uuid4()
    m.embedding_provider = "deterministic_hash"
    m.embedding_dimensions = 16
    m.embedding_model = None
    m.manifest_hash = "c" * 64
    m.include_sparse = True
    return m


def test_dense_maps_distance_to_score(manifest) -> None:
    cid = uuid4()
    did = uuid4()
    chunk = MagicMock()
    chunk.id = cid
    chunk.document_id = did
    chunk.chunk_text = "body"
    chunk.heading = None
    chunk.chunk_index = 0
    chunk.chunking_strategy = "fixed_size"
    doc = MagicMock()
    doc.source_path = "p.md"
    doc.title = None

    with patch("app.retrieval.dense.ChunkEmbeddingRepository") as MockEmb:
        inst = MockEmb.return_value
        inst.find_similar.return_value = [(chunk, doc, 1.0)]
        ret = DenseRetriever(MagicMock())
        cfg = RetrievalConfig(mode="dense_only")
        out = ret.retrieve("hello", cfg, manifest)
    assert len(out) == 1
    assert out[0].chunk_id == cid
    assert abs(out[0].dense_score - 1.0 / 2.0) < 1e-9
    assert out[0].metadata["dense_distance"] == 1.0


def test_dense_empty_query(manifest) -> None:
    ret = DenseRetriever(MagicMock())
    with pytest.raises(EmptyQueryError):
        ret.retrieve("   ", RetrievalConfig(), manifest)


def test_dense_returns_empty_when_no_rows(manifest) -> None:
    with patch("app.retrieval.dense.ChunkEmbeddingRepository") as MockEmb:
        MockEmb.return_value.find_similar.return_value = []
        ret = DenseRetriever(MagicMock())
        out = ret.retrieve("q", RetrievalConfig(), manifest)
    assert out == []
