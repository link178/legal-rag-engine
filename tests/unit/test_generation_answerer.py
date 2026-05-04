"""GroundedAnswerer orchestration (mocked retrieval / providers)."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from app.generation.answerer import GroundedAnswerer
from app.generation.context import ContextBuilder
from app.generation.providers.base import INSUFFICIENT_CONTEXT_SENTENCE
from app.generation.providers.mock import MockGenerationProvider
from app.retrieval.errors import EmptyQueryError
from app.retrieval.models import RetrievalConfig, RetrievalResultSet, RetrievedChunk


def _chunk(text: str, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        text=text,
        source_path="p.md",
        title=None,
        heading=None,
        chunk_index=0,
        chunking_strategy="fixed_size",
        dense_score=0.5,
        sparse_score=None,
        rrf_score=None,
        rank_position=rank,
        retrieval_sources=("dense",),
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


class _EchoProvider:
    name = "echo"

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self._reply


def _answerer_without_session(
    orchestrator,
    manifest,
    *,
    provider=_EchoProvider("x"),
    builder: ContextBuilder | None = None,
) -> GroundedAnswerer:
    cfg = RetrievalConfig(mode="dense_only")
    return GroundedAnswerer(
        orchestrator=orchestrator,
        manifest=manifest,
        config=cfg,
        context_builder=builder or ContextBuilder(),
        provider=provider,
    )


def test_empty_chunks_skips_provider(manifest) -> None:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="q",
        mode="dense_only",
        results=[],
        manifest_hash="h" * 64,
    )
    prov = _EchoProvider("should not run")
    g = _answerer_without_session(orch, manifest, provider=prov)
    out = g.answer(" hello ")
    assert out.mode == "insufficient_context"
    assert prov.calls == 0
    assert out.answer == INSUFFICIENT_CONTEXT_SENTENCE.strip()
    orch.retrieve.assert_called_once()


def test_grounded_with_valid_brackets(manifest) -> None:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="q",
        mode="hybrid",
        results=[_chunk("evidence one")],
        manifest_hash="a" * 64,
    )
    prov = _EchoProvider("Summary with support [99] invalid detail [1] valid.")
    g = _answerer_without_session(orch, manifest, provider=prov)
    out = g.answer("q")
    assert out.mode == "grounded"
    assert out.used_citation_ids == (1,)
    assert len(out.citations) == 1
    assert prov.calls == 1
    md = out.metadata
    assert md["generation_provider"] == "echo"
    assert md["retrieval_mode"] == "hybrid"
    assert md["total_retrieved_chunks"] == 1
    assert md["total_context_blocks"] == 1


def test_partial_when_no_brackets(manifest) -> None:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="q",
        mode="dense_only",
        results=[_chunk("body")],
    )
    prov = _EchoProvider("no citations here.")
    out = _answerer_without_session(orch, manifest, provider=prov).answer("q")
    assert out.mode == "partial"
    assert out.used_citation_ids == ()


def test_partial_invalid_bracket_only(manifest) -> None:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="q",
        mode="dense_only",
        results=[_chunk("body")],
    )
    prov = _EchoProvider("Only [99] which is invalid.")
    out = _answerer_without_session(orch, manifest, provider=prov).answer("q")
    assert out.mode == "partial"
    assert out.used_citation_ids == ()


def test_insufficient_when_provider_returns_sentinel(manifest) -> None:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="q",
        mode="dense_only",
        results=[_chunk("maybe")],
    )
    prov = _EchoProvider(INSUFFICIENT_CONTEXT_SENTENCE)
    out = _answerer_without_session(orch, manifest, provider=prov).answer("q")
    assert out.mode == "insufficient_context"
    assert out.citations == ()
    assert out.used_citation_ids == ()


def test_empty_question(manifest) -> None:
    orch = MagicMock()
    g = _answerer_without_session(orch, manifest, provider=MockGenerationProvider())
    with pytest.raises(EmptyQueryError):
        g.answer("  ")


def test_mock_integration_path(manifest) -> None:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="x",
        mode="dense_only",
        results=[_chunk("snippet")],
    )
    ga = GroundedAnswerer(
        orchestrator=orch,
        manifest=manifest,
        config=RetrievalConfig(mode="dense_only"),
        context_builder=ContextBuilder(),
        provider=MockGenerationProvider(),
    )
    out = ga.answer("question?")
    assert out.mode == "grounded"
