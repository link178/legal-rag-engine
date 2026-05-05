"""Evaluation CLI wiring (mocked DB path)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from app.evaluation.cli import main
from app.evaluation.models import RetrievalEvaluationItem, RetrievalEvaluationSummary
from app.retrieval.errors import ManifestNotFoundError


@pytest.fixture
def fake_settings():
    class _S:
        database_url = "postgresql://localhost/test"
        retrieval_mode = "hybrid"
        embedding_provider = "deterministic_hash"
        embedding_dimensions = 16
        embedding_model = ""

    return _S()


def _fake_summary() -> RetrievalEvaluationSummary:
    item = RetrievalEvaluationItem(
        question_id="q1",
        question="?",
        mode="hybrid",
        top_k=5,
        hit=True,
        hit_rank=1,
        reciprocal_rank=1.0,
        matched_by=("term",),
        retrieved_chunk_ids=(uuid4(),),
        retrieved_document_ids=(uuid4(),),
        retrieved_source_paths=("intro.md",),
        retrieved_scores=(0.5,),
        error=None,
    )
    return RetrievalEvaluationSummary(
        schema_version="evaluation.retrieval.v1",
        created_at=datetime(2026, 5, 4, tzinfo=UTC),
        config={"mode": "hybrid", "top_k": 5},
        manifest={"id": str(uuid4())},
        total_questions=1,
        answered_questions=1,
        errored_questions=0,
        hit_rate=1.0,
        mrr=1.0,
        items=(item,),
    )


def test_cli_json_stdout(fake_settings, capsys: pytest.CaptureFixture[str]) -> None:
    summary = _fake_summary()
    with patch("app.evaluation.cli.session_scope"):
        with patch("app.evaluation.cli.get_settings", return_value=fake_settings):
            with patch("app.evaluation.cli.RetrievalEvaluationRunner") as MockRunner:
                inst = MagicMock()
                inst.run_file.return_value = summary
                MockRunner.from_session.return_value = inst
                rc = main(["golden.jsonl", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == "evaluation.retrieval.v1"
    assert out["summary"]["total_questions"] == 1


def test_cli_writes_output_and_markdown(fake_settings, tmp_path: Path) -> None:
    summary = _fake_summary()
    gpath = tmp_path / "g.jsonl"
    gpath.write_text('{"id":"x","question":"?","expected_terms":["t"]}\n', encoding="utf-8")
    jout = tmp_path / "out.json"
    mout = tmp_path / "out.md"
    with patch("app.evaluation.cli.session_scope"):
        with patch("app.evaluation.cli.get_settings", return_value=fake_settings):
            with patch("app.evaluation.cli.RetrievalEvaluationRunner") as MockRunner:
                inst = MagicMock()
                inst.run_file.return_value = summary
                MockRunner.from_session.return_value = inst
                rc = main(
                    [
                        str(gpath),
                        "--output",
                        str(jout),
                        "--markdown-output",
                        str(mout),
                    ]
                )
    assert rc == 0
    assert jout.is_file()
    loaded = json.loads(jout.read_text(encoding="utf-8"))
    assert loaded["summary"]["mrr"] == 1.0
    md = mout.read_text(encoding="utf-8")
    assert "Retrieval evaluation report" in md
    assert "q1" in md


def test_cli_manifest_not_found(fake_settings, capsys: pytest.CaptureFixture[str]) -> None:
    with patch("app.evaluation.cli.session_scope"):
        with patch(
            "app.evaluation.cli.RetrievalEvaluationRunner.from_session",
            side_effect=ManifestNotFoundError("none"),
        ):
            with patch("app.evaluation.cli.get_settings", return_value=fake_settings):
                rc = main(["x.jsonl"])
    assert rc == 1
    assert "indexing.cli" in capsys.readouterr().err


def test_cli_invalid_manifest_uuid(fake_settings) -> None:
    with patch("app.evaluation.cli.get_settings", return_value=fake_settings):
        rc = main(["g.jsonl", "--index-manifest-id", "not-a-uuid"])
    assert rc == 1


def test_retrieval_cli_metadata_filter(fake_settings, tmp_path: Path) -> None:
    from app.retrieval.models import RetrievalMetadataFilter

    summary = _fake_summary()
    gpath = tmp_path / "g.jsonl"
    gpath.write_text('{"id":"x","question":"?","expected_terms":["t"]}\n', encoding="utf-8")
    with patch("app.evaluation.cli.session_scope"):
        with patch("app.evaluation.cli.get_settings", return_value=fake_settings):
            with patch("app.evaluation.cli.RetrievalEvaluationRunner") as MockRunner:
                inst = MagicMock()
                inst.run_file.return_value = summary
                MockRunner.from_session.return_value = inst
                rc = main(["retrieval", str(gpath), "--filter-jurisdiction", "eu"])
    assert rc == 0
    cfg = MockRunner.from_session.call_args[0][1]
    assert cfg.metadata_filter == RetrievalMetadataFilter(jurisdiction="eu")
