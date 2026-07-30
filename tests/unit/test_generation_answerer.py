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


def _chunk(
    text: str,
    rank: int = 1,
    *,
    source_path: str = "docs/evidence.md",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        text=text,
        source_path=source_path,
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
    assert out.citation_verification is not None
    assert out.citation_verification.has_citations is False
    assert out.citation_verification.available_citation_ids == ()
    assert out.metadata.get("citation_validity_rate") == 0.0
    assert out.metadata.get("evidence_sufficient") is False
    orch.retrieve.assert_called_once()


def test_grounded_with_valid_brackets(manifest) -> None:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="evidence summary question",
        mode="hybrid",
        results=[_chunk("evidence one summary")],
        manifest_hash="a" * 64,
    )
    prov = _EchoProvider("Summary supported by [1].")
    g = _answerer_without_session(orch, manifest, provider=prov)
    out = g.answer("evidence summary question")
    assert out.mode == "grounded"
    assert out.used_citation_ids == (1,)
    assert len(out.citations) == 1
    assert prov.calls == 1
    assert out.citation_verification is not None
    assert out.citation_verification.has_invalid_citations is False
    assert out.metadata.get("evidence_sufficient") is True
    md = out.metadata
    assert md["generation_provider"] == "echo"
    assert md["retrieval_mode"] == "hybrid"
    assert md["total_retrieved_chunks"] == 1
    assert md["total_context_blocks"] == 1
    assert md["citation_validity_rate"] == 1.0


def test_partial_when_mixed_valid_and_invalid(manifest) -> None:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="evidence summary",
        mode="hybrid",
        results=[_chunk("evidence one summary")],
        manifest_hash="a" * 64,
    )
    prov = _EchoProvider("Summary with support [99] invalid detail [1] valid.")
    out = _answerer_without_session(orch, manifest, provider=prov).answer(
        "evidence summary"
    )
    assert out.mode == "partial"
    assert out.used_citation_ids == (1,)
    cv = out.citation_verification
    assert cv is not None
    assert cv.has_valid_citations is True
    assert cv.has_invalid_citations is True


def test_partial_when_no_brackets(manifest) -> None:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="body content",
        mode="dense_only",
        results=[_chunk("body content here")],
    )
    prov = _EchoProvider("no citations here.")
    out = _answerer_without_session(orch, manifest, provider=prov).answer("body content")
    assert out.mode == "partial"
    assert out.used_citation_ids == ()
    cv = out.citation_verification
    assert cv is not None
    assert cv.has_citations is False


def test_partial_invalid_bracket_only(manifest) -> None:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="body content",
        mode="dense_only",
        results=[_chunk("body content here")],
    )
    prov = _EchoProvider("Only [99] which is invalid.")
    out = _answerer_without_session(orch, manifest, provider=prov).answer("body content")
    assert out.mode == "partial"
    assert out.used_citation_ids == ()
    cv = out.citation_verification
    assert cv is not None
    assert cv.has_invalid_citations is True


def test_insufficient_when_provider_returns_sentinel(manifest) -> None:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="maybe evidence",
        mode="dense_only",
        results=[_chunk("maybe evidence text")],
    )
    prov = _EchoProvider(INSUFFICIENT_CONTEXT_SENTENCE)
    out = _answerer_without_session(orch, manifest, provider=prov).answer(
        "maybe evidence"
    )
    assert out.mode == "insufficient_context"
    assert out.citations == ()
    assert out.used_citation_ids == ()
    assert prov.calls == 1
    cv = out.citation_verification
    assert cv is not None
    assert cv.has_citations is False
    assert cv.available_citation_ids == (1,)


def test_irrelevant_retrieved_context_is_insufficient(manifest) -> None:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="What is the capital of Atlantis according to the corpus?",
        mode="sparse_only",
        results=[
            _chunk(
                "Hello from the sample corpus.\nThis file is plain UTF-8 text.",
                source_path="data/sample_corpus/basic/plain.txt",
            )
        ],
    )
    prov = _EchoProvider("Based on the provided context [1]")
    out = _answerer_without_session(orch, manifest, provider=prov).answer(
        "What is the capital of Atlantis according to the corpus?"
    )
    assert out.mode == "insufficient_context"
    assert prov.calls == 0
    assert out.metadata.get("evidence_sufficient") is False


def test_valid_citations_over_irrelevant_context_not_grounded(manifest) -> None:
    """Answerer must not classify grounded merely because citations are valid."""
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="atomic weight of Unobtainium-99",
        mode="hybrid",
        results=[_chunk("Markdown sample for ingestion tests.", source_path="intro.md")],
    )
    # Would be grounded under citation-only rules if the evidence gate were skipped.
    prov = _EchoProvider("The answer is 42 [1].")
    out = _answerer_without_session(orch, manifest, provider=prov).answer(
        "atomic weight of Unobtainium-99"
    )
    assert out.mode == "insufficient_context"
    assert prov.calls == 0


def test_relevant_evidence_grounded_with_mock(manifest) -> None:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="What does the basic intro file describe?",
        mode="sparse_only",
        results=[
            _chunk(
                "# Sample intro\nShort Markdown sample for ingestion.",
                source_path="data/sample_corpus/basic/intro.md",
            )
        ],
    )
    ga = GroundedAnswerer(
        orchestrator=orch,
        manifest=manifest,
        config=RetrievalConfig(mode="sparse_only"),
        context_builder=ContextBuilder(),
        provider=MockGenerationProvider(),
    )
    out = ga.answer("What does the basic intro file describe?")
    assert out.mode == "grounded"
    assert out.used_citation_ids == (1,)


def test_empty_question(manifest) -> None:
    orch = MagicMock()
    g = _answerer_without_session(orch, manifest, provider=MockGenerationProvider())
    with pytest.raises(EmptyQueryError):
        g.answer("  ")


def test_mock_integration_path(manifest) -> None:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="snippet evidence",
        mode="dense_only",
        results=[_chunk("snippet evidence text")],
    )
    ga = GroundedAnswerer(
        orchestrator=orch,
        manifest=manifest,
        config=RetrievalConfig(mode="dense_only"),
        context_builder=ContextBuilder(),
        provider=MockGenerationProvider(),
    )
    out = ga.answer("snippet evidence")
    assert out.mode == "grounded"
    assert out.citation_verification is not None
    assert out.citation_verification.citation_validity_rate == 1.0


def test_metadata_filter_echoed_in_answer_metadata(manifest) -> None:
    from app.retrieval.models import RetrievalMetadataFilter

    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="snippet evidence",
        mode="dense_only",
        results=[_chunk("snippet evidence text")],
    )
    ga = GroundedAnswerer(
        orchestrator=orch,
        manifest=manifest,
        config=RetrievalConfig(
            mode="dense_only",
            metadata_filter=RetrievalMetadataFilter(jurisdiction="eu"),
        ),
        context_builder=ContextBuilder(),
        provider=MockGenerationProvider(),
    )
    out = ga.answer("snippet evidence")
    assert out.metadata.get("metadata_filter") == {"jurisdiction": "eu"}
