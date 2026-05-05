"""AnswerEvaluationRunner orchestration (mocked answerer)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from app.evaluation.models import AnswerGoldenQuestion
from app.evaluation.runners.answer import AnswerEvaluationRunner, load_answer_golden
from app.generation.context import ContextBuilder
from app.generation.models import CitationVerificationResult, GroundedAnswer, GroundedCitation
from app.generation.providers.mock import MockGenerationProvider
from app.retrieval.errors import EmptyQueryError
from app.retrieval.models import RetrievalConfig, RetrievalMetadataFilter
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


def _citation(cid: int, preview: str, path: str | None = "intro.md") -> GroundedCitation:
    return GroundedCitation(
        citation_id=cid,
        chunk_id=uuid4(),
        document_id=uuid4(),
        source_path=path,
        title=None,
        heading=None,
        rank=cid,
        score=0.5,
        text_preview=preview,
    )


def _simple_grounded() -> GroundedAnswer:
    cv = CitationVerificationResult(
        used_citation_ids=(1,),
        available_citation_ids=(1,),
        valid_citation_ids=(1,),
        invalid_citation_ids=(),
        unused_citation_ids=(),
        duplicate_citation_ids=(),
        citation_validity_rate=1.0,
        has_citations=True,
        has_valid_citations=True,
        has_invalid_citations=False,
    )
    return GroundedAnswer(
        question="q",
        answer="Based on the provided context [1].",
        mode="grounded",
        citations=(_citation(1, "markdown sample", "intro.md"),),
        used_citation_ids=(1,),
        retrieval_mode="hybrid",
        insufficient_context=False,
        citation_verification=cv,
    )


def test_load_answer_golden_rejects_duplicate_id(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text(
        '{"id":"a","question":"x?"}\n{"id":"a","question":"y?"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_answer_golden(path)


def test_load_answer_golden_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_answer_golden(path)


def test_answer_runner_multiple_questions() -> None:
    cfg = RetrievalConfig(mode="hybrid", top_k=5)
    mf = _manifest()
    ga = MagicMock()
    ga.answer.side_effect = [_simple_grounded(), _simple_grounded()]
    cb = ContextBuilder()
    runner = AnswerEvaluationRunner(
        ga,
        mf,
        cfg,
        context_builder=cb,
        generation_provider_name=MockGenerationProvider.name,
    )
    qs = (
        AnswerGoldenQuestion(
            id="q1",
            question="one?",
            expected_mode="grounded",
            expected_terms=("markdown",),
            expected_source_paths=("intro.md",),
        ),
        AnswerGoldenQuestion(
            id="q2",
            question="two?",
            expected_mode="grounded",
            expected_terms=("markdown",),
            expected_source_paths=("intro.md",),
        ),
    )
    summary = runner.run_questions(qs, top_k=5)
    assert summary.total_questions == 2
    assert summary.answered_questions == 2
    assert summary.errored_questions == 0
    assert ga.answer.call_count == 2
    assert all(i.passed for i in summary.items)


def test_answer_runner_retrieval_error_continues() -> None:
    cfg = RetrievalConfig(mode="hybrid", top_k=5)
    mf = _manifest()
    ga = MagicMock()

    def side_effect(q: str):
        if q == "bad":
            raise EmptyQueryError("empty")
        return _simple_grounded()

    ga.answer.side_effect = side_effect
    runner = AnswerEvaluationRunner(
        ga,
        mf,
        cfg,
        context_builder=ContextBuilder(),
        generation_provider_name="mock",
    )
    qs = (
        AnswerGoldenQuestion(
            id="a",
            question="bad",
            expected_mode="grounded",
            expected_terms=("x",),
        ),
        AnswerGoldenQuestion(
            id="b",
            question="ok?",
            expected_mode="grounded",
            expected_terms=("markdown",),
            expected_source_paths=("intro.md",),
        ),
    )
    summary = runner.run_questions(qs, top_k=5)
    assert summary.errored_questions == 1
    assert summary.answered_questions == 1
    assert any(i.error for i in summary.items)
    assert any(not i.error for i in summary.items)


def test_answer_runner_warns_recommended_mode_mismatch(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = RetrievalConfig(mode="hybrid", top_k=5)
    mf = _manifest()
    ga = MagicMock()
    ga.answer.return_value = _simple_grounded()
    runner = AnswerEvaluationRunner(
        ga,
        mf,
        cfg,
        context_builder=ContextBuilder(),
        generation_provider_name="mock",
    )
    g = AnswerGoldenQuestion(
        id="w",
        question="?",
        recommended_mode="sparse_only",
        expected_mode="grounded",
        expected_terms=("markdown",),
        expected_source_paths=("intro.md",),
    )
    runner.run_questions((g,), top_k=5)
    err = capsys.readouterr().err
    assert "warn:" in err and "sparse_only" in err


def test_answer_runner_summary_includes_metadata_filter() -> None:
    mf = RetrievalMetadataFilter(jurisdiction="eu")
    cfg = RetrievalConfig(mode="hybrid", top_k=5, metadata_filter=mf)
    mf_row = _manifest()
    ga = MagicMock()
    ga.answer.return_value = _simple_grounded()
    runner = AnswerEvaluationRunner(
        ga,
        mf_row,
        cfg,
        context_builder=ContextBuilder(),
        generation_provider_name="mock",
    )
    g = AnswerGoldenQuestion(
        id="q1",
        question="one?",
        expected_mode="grounded",
        expected_terms=("markdown",),
        expected_source_paths=("intro.md",),
    )
    summary = runner.run_questions((g,), top_k=3)
    assert summary.config.get("metadata_filter") == {"jurisdiction": "eu"}
