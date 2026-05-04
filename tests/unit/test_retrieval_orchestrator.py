"""RetrievalOrchestrator modes and validation."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from app.retrieval.errors import EmptyQueryError, RetrieverNotConfiguredError
from app.retrieval.models import RetrievalConfig, RetrievedChunk
from app.retrieval.orchestrator import RetrievalOrchestrator


def _one_hit(suffix: str = "a") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        text=f"t{suffix}",
        source_path=None,
        title=None,
        heading=None,
        chunk_index=0,
        chunking_strategy="fixed_size",
        dense_score=0.9 if suffix == "d" else None,
        sparse_score=0.8 if suffix == "s" else None,
        rank_position=1,
        retrieval_sources=("dense",) if suffix == "d" else ("sparse",),
    )


@pytest.fixture
def manifest():
    m = MagicMock()
    m.id = uuid4()
    m.manifest_hash = "m" * 64
    m.embedding_provider = "deterministic_hash"
    m.embedding_model = None
    m.embedding_dimensions = 16
    m.include_sparse = True
    return m


def test_dense_only_requires_dense(manifest) -> None:
    s = MagicMock()
    s.retrieve.return_value = [_one_hit("d")]
    orch = RetrievalOrchestrator(None, s)
    with pytest.raises(RetrieverNotConfiguredError, match="dense"):
        orch.retrieve("q", RetrievalConfig(mode="dense_only"), manifest)


def test_sparse_only_requires_sparse(manifest) -> None:
    d = MagicMock()
    d.retrieve.return_value = [_one_hit("d")]
    orch = RetrievalOrchestrator(d, None)
    with pytest.raises(RetrieverNotConfiguredError, match="sparse"):
        orch.retrieve("q", RetrievalConfig(mode="sparse_only"), manifest)


def test_hybrid_requires_both(manifest) -> None:
    orch = RetrievalOrchestrator(MagicMock(), None)
    with pytest.raises(RetrieverNotConfiguredError, match="both"):
        orch.retrieve("q", RetrievalConfig(mode="hybrid"), manifest)


def test_empty_query(manifest) -> None:
    orch = RetrievalOrchestrator(MagicMock(), MagicMock())
    with pytest.raises(EmptyQueryError):
        orch.retrieve("  ", RetrievalConfig(), manifest)


def test_dense_only_calls_dense(manifest) -> None:
    d = MagicMock()
    d.retrieve.return_value = [_one_hit("d")]
    orch = RetrievalOrchestrator(d, MagicMock())
    res = orch.retrieve("hello", RetrievalConfig(mode="dense_only", top_k=5), manifest)
    d.retrieve.assert_called_once()
    assert res.results[0].rrf_score is None


def test_hybrid_fuses(manifest) -> None:
    d = MagicMock()
    s = MagicMock()
    d.retrieve.return_value = [_one_hit("d")]
    s.retrieve.return_value = [_one_hit("s")]
    orch = RetrievalOrchestrator(d, s)
    res = orch.retrieve("hello", RetrievalConfig(mode="hybrid", top_k=5), manifest)
    d.retrieve.assert_called_once()
    s.retrieve.assert_called_once()
    assert len(res.results) >= 1
    assert res.results[0].rrf_score is not None


def test_hybrid_empty_dense_still_rrf(manifest) -> None:
    d = MagicMock()
    s = MagicMock()
    d.retrieve.return_value = []
    s.retrieve.return_value = [_one_hit("s")]
    orch = RetrievalOrchestrator(d, s)
    res = orch.retrieve("hello", RetrievalConfig(mode="hybrid"), manifest)
    assert len(res.results) == 1
    assert res.results[0].rrf_score is not None


def test_hybrid_both_empty(manifest) -> None:
    d = MagicMock()
    s = MagicMock()
    d.retrieve.return_value = []
    s.retrieve.return_value = []
    orch = RetrievalOrchestrator(d, s)
    res = orch.retrieve("hello", RetrievalConfig(mode="hybrid"), manifest)
    assert res.results == []


def test_hybrid_rejects_manifest_without_sparse(manifest) -> None:
    manifest.include_sparse = False
    d = MagicMock()
    s = MagicMock()
    d.retrieve.return_value = []
    s.retrieve.return_value = []
    orch = RetrievalOrchestrator(d, s)
    from app.retrieval.errors import ManifestNotFoundError

    with pytest.raises(ManifestNotFoundError, match="sparse"):
        orch.retrieve("hello", RetrievalConfig(mode="hybrid"), manifest)
