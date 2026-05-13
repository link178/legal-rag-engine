"""RetrievalEvaluationRunner orchestration (mocked retrieval)."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from app.evaluation.models import RetrievalGoldenQuestion
from app.evaluation.runners.retrieval import RetrievalEvaluationRunner
from app.retrieval.errors import EmptyQueryError
from app.retrieval.models import (
    RetrievalConfig,
    RetrievalMetadataFilter,
    RetrievalResultSet,
    RetrievedChunk,
)
from app.storage.postgres.models import IndexManifestRecord


def _manifest() -> MagicMock:
    m = MagicMock(spec=IndexManifestRecord)
    m.id = uuid4()
    m.manifest_hash = "h" * 64
    m.include_sparse = True
    m.embeddings_persisted = True
    m.embedding_provider = "deterministic_hash"
    m.embedding_model = None
    m.embedding_dimensions = 16
    m.chunking_strategy = "fixed_size"
    return m


def _orch_with_hits(hits: list[RetrievedChunk]) -> MagicMock:
    orch = MagicMock()
    orch.retrieve.return_value = RetrievalResultSet(
        query="q",
        mode="hybrid",
        results=hits,
        index_manifest_id=uuid4(),
        manifest_hash="x" * 64,
    )
    return orch


def _one_hit() -> RetrievedChunk:
    cid = uuid4()
    did = uuid4()
    return RetrievedChunk(
        chunk_id=cid,
        document_id=did,
        text="foo bar baz",
        source_path="intro.md",
        title=None,
        heading=None,
        chunk_index=0,
        chunking_strategy="fixed_size",
        dense_score=0.9,
        rank_position=1,
        retrieval_sources=("dense",),
    )


def test_runner_processes_multiple_questions() -> None:
    cfg = RetrievalConfig(mode="hybrid", top_k=3)
    mf = _manifest()
    h = [_one_hit()]
    orch = _orch_with_hits(h)
    session = MagicMock()
    runner = RetrievalEvaluationRunner(session, orch, mf, cfg)
    qs = (
        RetrievalGoldenQuestion(id="q1", question="one?", expected_terms=("foo",)),
        RetrievalGoldenQuestion(id="q2", question="two?", expected_terms=("foo",)),
    )
    summary = runner.run_questions(qs, top_k=3)
    assert summary.total_questions == 2
    assert summary.answered_questions == 2
    assert summary.errored_questions == 0
    assert orch.retrieve.call_count == 2


def test_runner_one_error_continues() -> None:
    cfg = RetrievalConfig(mode="hybrid", top_k=3)
    mf = _manifest()
    orch = MagicMock()

    def side_effect(query: str, _cfg, _mf):
        if query == "bad":
            raise EmptyQueryError("empty")
        return RetrievalResultSet(query=query, mode="hybrid", results=[_one_hit()])

    orch.retrieve.side_effect = side_effect
    session = MagicMock()
    runner = RetrievalEvaluationRunner(session, orch, mf, cfg)
    qs = (
        RetrievalGoldenQuestion(id="a", question="bad", expected_terms=("x",)),
        RetrievalGoldenQuestion(id="b", question="ok?", expected_terms=("foo",)),
    )
    summary = runner.run_questions(qs, top_k=3)
    assert summary.errored_questions == 1
    assert summary.answered_questions == 1
    assert any(i.error for i in summary.items)
    assert any(not i.error for i in summary.items)


def test_runner_respects_top_k_in_grading() -> None:
    cfg = RetrievalConfig(mode="hybrid", top_k=1)
    mf = _manifest()
    c1 = _one_hit()
    c2 = RetrievedChunk(
        chunk_id=uuid4(),
        document_id=c1.document_id,
        text="has targetterm herexx",
        source_path="other.md",
        title=None,
        heading=None,
        chunk_index=1,
        chunking_strategy="fixed_size",
        dense_score=0.8,
        rank_position=2,
        retrieval_sources=("dense",),
    )
    orch = _orch_with_hits([c1, c2])
    session = MagicMock()
    runner = RetrievalEvaluationRunner(session, orch, mf, cfg)
    g = RetrievalGoldenQuestion(
        id="q",
        question="?",
        expected_terms=("targetterm",),
    )
    summary = runner.run_questions((g,), top_k=1)
    assert summary.items[0].hit is False
    summary2 = runner.run_questions((g,), top_k=2)
    assert summary2.items[0].hit is True
    assert summary2.items[0].hit_rank == 2


def test_runner_preserves_metadata_filter_in_rebuilt_config() -> None:
    meta_f = RetrievalMetadataFilter(jurisdiction="eu")
    cfg = RetrievalConfig(mode="hybrid", top_k=3, metadata_filter=meta_f)
    manifest = _manifest()
    orch = _orch_with_hits([_one_hit()])
    session = MagicMock()
    runner = RetrievalEvaluationRunner(session, orch, manifest, cfg)
    qs = (RetrievalGoldenQuestion(id="q1", question="one?", expected_terms=("foo",)),)
    summary = runner.run_questions(qs, top_k=7)
    assert summary.config.get("metadata_filter") == {"jurisdiction": "eu"}
    call_cfg = orch.retrieve.call_args.args[1]
    assert call_cfg.top_k == 7
    assert call_cfg.metadata_filter == meta_f
