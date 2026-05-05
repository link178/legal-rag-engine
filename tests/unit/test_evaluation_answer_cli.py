"""Answer evaluation CLI + legacy retrieval dispatch."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from app.evaluation.cli import main
from app.evaluation.models import (
    AnswerEvaluationItem,
    AnswerEvaluationSummary,
    RetrievalEvaluationItem,
    RetrievalEvaluationSummary,
)
from app.retrieval.errors import ManifestNotFoundError


@pytest.fixture
def fake_settings():
    class _S:
        database_url = "postgresql://localhost/test"
        retrieval_mode = "hybrid"
        embedding_provider = "deterministic_hash"
        embedding_dimensions = 16
        embedding_model = ""
        generation_provider = "mock"

    return _S()


def _answer_summary() -> AnswerEvaluationSummary:
    item = AnswerEvaluationItem(
        question_id="q1",
        question="?",
        mode="hybrid",
        answer_mode="grounded",
        expected_mode="grounded",
        mode_matches=True,
        contains_expected_terms=True,
        missing_expected_terms=(),
        citation_validity_rate=1.0,
        has_valid_citations=True,
        has_invalid_citations=False,
        insufficient_context_matches=True,
        retrieved_expected_source=True,
        cited_source_paths=("intro.md",),
        used_citation_ids=(1,),
        invalid_citation_ids=(),
        passed=True,
        error=None,
    )
    return AnswerEvaluationSummary(
        schema_version="evaluation.answer.v1",
        created_at=datetime(2026, 5, 5, tzinfo=UTC),
        config={"mode": "hybrid", "top_k": 5, "provider": "mock"},
        manifest={"id": str(uuid4())},
        total_questions=1,
        answered_questions=1,
        errored_questions=0,
        pass_rate=1.0,
        mode_accuracy=1.0,
        expected_terms_accuracy=1.0,
        citation_validity_rate_avg=1.0,
        insufficient_context_accuracy=1.0,
        retrieved_expected_source_rate=1.0,
        items=(item,),
    )


def tmp_golden(content: str) -> Path:
    import tempfile

    p = Path(tempfile.mkdtemp()) / "a.jsonl"
    p.write_text(content.strip() + "\n", encoding="utf-8")
    return p


def test_answer_cli_json_stdout(fake_settings, capsys: pytest.CaptureFixture[str]) -> None:
    summary = _answer_summary()
    gpath = tmp_golden("""
{"id":"x","question":"?","expected_mode":"grounded","expected_terms":["t"]}
""")

    inst = MagicMock()
    inst.run_file.return_value = summary

    with patch("app.evaluation.cli.session_scope"):
        with patch("app.evaluation.cli.get_settings", return_value=fake_settings):
            with patch(
                "app.evaluation.cli.AnswerEvaluationRunner.from_session",
                return_value=inst,
            ):
                rc = main(["answer", str(gpath), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == "evaluation.answer.v1"
    assert out["summary"]["pass_rate"] == 1.0


def test_answer_cli_rejects_non_mock_provider() -> None:
    class _Bad:
        database_url = "postgresql://localhost/test"
        retrieval_mode = "hybrid"
        embedding_provider = "deterministic_hash"
        embedding_dimensions = 16
        embedding_model = ""
        generation_provider = "openai"

    class _Good:
        database_url = "postgresql://localhost/test"
        retrieval_mode = "hybrid"
        embedding_provider = "deterministic_hash"
        embedding_dimensions = 16
        embedding_model = ""
        generation_provider = "mock"

    g = tmp_golden('{"id":"x","question":"?","expected_terms":["z"]}')
    with patch("app.evaluation.cli.get_settings", return_value=_Bad()):
        rc = main(["answer", str(g)])
    assert rc == 1

    g2 = tmp_golden('{"id":"y","question":"?","expected_terms":["z"]}')
    with patch("app.evaluation.cli.get_settings", return_value=_Good()):
        rc = main(["answer", str(g2), "--provider", "ollama"])
    assert rc == 1


def test_retrieval_legacy_dispatch(fake_settings, capsys: pytest.CaptureFixture[str]) -> None:
    rid = uuid4()
    item = RetrievalEvaluationItem(
        question_id="q1",
        question="?",
        mode="hybrid",
        top_k=5,
        hit=True,
        hit_rank=1,
        reciprocal_rank=1.0,
        matched_by=("term",),
        retrieved_chunk_ids=(rid,),
        retrieved_document_ids=(uuid4(),),
        retrieved_source_paths=("intro.md",),
        retrieved_scores=(0.5,),
        error=None,
    )
    summary = RetrievalEvaluationSummary(
        schema_version="evaluation.retrieval.v1",
        created_at=datetime(2026, 5, 5, tzinfo=UTC),
        config={"mode": "hybrid", "top_k": 5},
        manifest={"id": str(uuid4())},
        total_questions=1,
        answered_questions=1,
        errored_questions=0,
        hit_rate=1.0,
        mrr=1.0,
        items=(item,),
    )
    gpath = tmp_golden('{"id":"x","question":"?","expected_terms":["t"]}')
    with patch("app.evaluation.cli.session_scope"):
        with patch("app.evaluation.cli.get_settings", return_value=fake_settings):
            with patch("app.evaluation.cli.RetrievalEvaluationRunner") as MockRunner:
                inst = MagicMock()
                inst.run_file.return_value = summary
                MockRunner.from_session.return_value = inst
                rc = main([str(gpath), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == "evaluation.retrieval.v1"


def test_retrieval_explicit_subcommand(fake_settings, capsys: pytest.CaptureFixture[str]) -> None:
    rid = uuid4()
    item = RetrievalEvaluationItem(
        question_id="q1",
        question="?",
        mode="hybrid",
        top_k=5,
        hit=True,
        hit_rank=1,
        reciprocal_rank=1.0,
        matched_by=("term",),
        retrieved_chunk_ids=(rid,),
        retrieved_document_ids=(uuid4(),),
        retrieved_source_paths=("intro.md",),
        retrieved_scores=(0.5,),
        error=None,
    )
    summary = RetrievalEvaluationSummary(
        schema_version="evaluation.retrieval.v1",
        created_at=datetime(2026, 5, 5, tzinfo=UTC),
        config={"mode": "hybrid", "top_k": 5},
        manifest={"id": str(uuid4())},
        total_questions=1,
        answered_questions=1,
        errored_questions=0,
        hit_rate=1.0,
        mrr=1.0,
        items=(item,),
    )
    gpath = tmp_golden('{"id":"x","question":"?","expected_terms":["t"]}')
    with patch("app.evaluation.cli.session_scope"):
        with patch("app.evaluation.cli.get_settings", return_value=fake_settings):
            with patch("app.evaluation.cli.RetrievalEvaluationRunner") as MockRunner:
                inst = MagicMock()
                inst.run_file.return_value = summary
                MockRunner.from_session.return_value = inst
                rc = main(["retrieval", str(gpath), "--json"])
    assert rc == 0


def test_answer_cli_manifest_not_found(fake_settings) -> None:
    g = tmp_golden('{"id":"x","question":"?","expected_terms":["z"]}')
    with patch("app.evaluation.cli.session_scope"):
        with patch(
            "app.evaluation.cli.AnswerEvaluationRunner.from_session",
            side_effect=ManifestNotFoundError("none"),
        ):
            with patch("app.evaluation.cli.get_settings", return_value=fake_settings):
                rc = main(["answer", str(g)])
    assert rc == 1


def test_answer_cli_writes_output(fake_settings, tmp_path: Path) -> None:
    summary = _answer_summary()
    gpath = tmp_path / "g.jsonl"
    gpath.write_text(
        '{"id":"x","question":"?","expected_mode":"grounded","expected_terms":["t"]}\n',
        encoding="utf-8",
    )
    jout = tmp_path / "out.json"
    mout = tmp_path / "out.md"
    inst = MagicMock()
    inst.run_file.return_value = summary

    with patch("app.evaluation.cli.session_scope"):
        with patch("app.evaluation.cli.get_settings", return_value=fake_settings):
            with patch(
                "app.evaluation.cli.AnswerEvaluationRunner.from_session",
                return_value=inst,
            ):
                rc = main(
                    [
                        "answer",
                        str(gpath),
                        "--output",
                        str(jout),
                        "--markdown-output",
                        str(mout),
                    ]
                )
    assert rc == 0
    loaded = json.loads(jout.read_text(encoding="utf-8"))
    assert loaded["summary"]["total_questions"] == 1
    md = mout.read_text(encoding="utf-8")
    assert "Answer evaluation report" in md
    assert "Failures" in md


def test_answer_cli_metadata_filter(fake_settings, tmp_path: Path) -> None:
    from app.retrieval.models import RetrievalMetadataFilter

    summary = _answer_summary()
    gpath = tmp_path / "g.jsonl"
    gpath.write_text(
        '{"id":"x","question":"?","expected_mode":"grounded","expected_terms":["t"]}\n',
        encoding="utf-8",
    )
    inst = MagicMock()
    inst.run_file.return_value = summary
    with patch("app.evaluation.cli.session_scope"):
        with patch("app.evaluation.cli.get_settings", return_value=fake_settings):
            with patch(
                "app.evaluation.cli.AnswerEvaluationRunner.from_session",
                return_value=inst,
            ) as MockFrom:
                rc = main(["answer", str(gpath), "--filter-jurisdiction", "eu"])
    assert rc == 0
    cfg = MockFrom.call_args[0][1]
    assert cfg.metadata_filter == RetrievalMetadataFilter(jurisdiction="eu")
