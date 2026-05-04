"""Generation CLI wiring (mocked DB session)."""

from __future__ import annotations

import json
from unittest.mock import patch
from uuid import uuid4

import pytest
from app.generation.cli import answer_to_dict, main
from app.generation.models import GroundedAnswer, GroundedCitation
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


def _sample_answer() -> GroundedAnswer:
    cid = uuid4()
    did = uuid4()
    cit = GroundedCitation(
        citation_id=1,
        chunk_id=cid,
        document_id=did,
        source_path="intro.md",
        title=None,
        heading=None,
        rank=1,
        score=0.25,
        text_preview="short",
    )
    return GroundedAnswer(
        question="q?",
        answer="reply [1]",
        mode="grounded",
        citations=(cit,),
        used_citation_ids=(1,),
        retrieval_mode="hybrid",
        insufficient_context=False,
        metadata={"total_context_blocks": 1},
    )


def test_answer_to_dict_serializable() -> None:
    ga = _sample_answer()
    d = answer_to_dict(ga)
    s = json.dumps(d)
    assert "chunk_id" in s and isinstance(d["citations"][0]["chunk_id"], str)


def test_cli_json_stdout(fake_settings, capsys: pytest.CaptureFixture[str]) -> None:
    ans = _sample_answer()
    with patch("app.generation.cli.session_scope"):
        with patch("app.generation.cli.GroundedAnswerer.from_session") as mf:
            mf.return_value.answer.return_value = ans
            with patch("app.generation.cli.get_settings", return_value=fake_settings):
                rc = main(["hello", "--provider", "mock", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "grounded"
    assert out["used_citation_ids"] == [1]


def test_cli_rejects_non_mock_provider(fake_settings) -> None:
    fake_bad = fake_settings
    fake_bad.generation_provider = "openai"
    with patch("app.generation.cli.get_settings", return_value=fake_bad):
        rc = main(["hello", "--json"])
    assert rc == 1


def test_cli_manifest_not_found(fake_settings, capsys: pytest.CaptureFixture[str]) -> None:
    with patch("app.generation.cli.session_scope"):
        with patch(
            "app.generation.cli.GroundedAnswerer.from_session",
            side_effect=ManifestNotFoundError("none"),
        ):
            with patch("app.generation.cli.get_settings", return_value=fake_settings):
                rc = main(["q", "--json", "--provider", "mock"])
    assert rc == 1
    assert "indexing.cli" in capsys.readouterr().err


def test_cli_default_mode_from_settings(fake_settings) -> None:
    fake_settings.retrieval_mode = "sparse_only"
    ans = _sample_answer()
    with patch("app.generation.cli.session_scope"):
        with patch("app.generation.cli.GroundedAnswerer.from_session") as mf:
            mf.return_value.answer.return_value = ans
            with patch("app.generation.cli.get_settings", return_value=fake_settings):
                rc = main(["x", "--provider", "mock"])
    assert rc == 0
    cfg = mf.call_args[0][1]
    assert cfg.mode == "sparse_only"


def test_cli_invalid_manifest_uuid(fake_settings) -> None:
    with patch("app.generation.cli.get_settings", return_value=fake_settings):
        rc = main(["q", "--index-manifest-id", "not-uuid", "--provider", "mock"])
    assert rc == 1
