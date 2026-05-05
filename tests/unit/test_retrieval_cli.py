"""Retrieval CLI wiring (mocked DB path)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from app.retrieval.cli import main
from app.retrieval.errors import ManifestNotFoundError
from app.retrieval.models import RetrievalMetadataFilter, RetrievalResultSet, RetrievedChunk


@pytest.fixture
def fake_settings():
    class _S:
        database_url = "postgresql://localhost/test"
        retrieval_mode = "hybrid"
        embedding_provider = "deterministic_hash"
        embedding_dimensions = 16
        embedding_model = ""

    return _S()


def test_cli_json_success(fake_settings, capsys: pytest.CaptureFixture[str]) -> None:
    mid = uuid4()
    ch = RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        text="hello world",
        source_path="p.md",
        title=None,
        heading=None,
        chunk_index=0,
        chunking_strategy="fixed_size",
        dense_score=0.5,
        sparse_score=0.3,
        rrf_score=0.02,
        rank_position=1,
        retrieval_sources=("dense", "sparse"),
    )
    res = RetrievalResultSet(
        query="q",
        mode="hybrid",
        results=[ch],
        index_manifest_id=mid,
        embedding_provider="deterministic_hash",
        embedding_dimensions=16,
        manifest_hash="a" * 64,
    )
    fake_manifest = MagicMock()
    fake_manifest.include_sparse = True

    with patch("app.retrieval.cli.session_scope"):
        with patch("app.retrieval.cli.resolve_manifest_record", return_value=fake_manifest):
            with patch("app.retrieval.cli.RetrievalOrchestrator") as MockOrch:
                MockOrch.return_value.retrieve.return_value = res
                with patch("app.retrieval.cli.get_settings", return_value=fake_settings):
                    rc = main(["hello", "--json", "--mode", "hybrid"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_results"] == 1
    assert payload["manifest_id"] == str(mid)
    assert payload["results"][0]["rank"] == 1
    assert payload["results"][0]["text_preview"] == "hello world"


def test_cli_default_mode_from_settings(fake_settings) -> None:
    fake_settings.retrieval_mode = "dense_only"
    res = RetrievalResultSet(query="q", mode="dense_only", results=[], manifest_hash="b" * 64)
    fake_manifest = MagicMock()
    fake_manifest.include_sparse = False

    with patch("app.retrieval.cli.session_scope"):
        with patch("app.retrieval.cli.resolve_manifest_record", return_value=fake_manifest):
            with patch("app.retrieval.cli.RetrievalOrchestrator") as MockOrch:
                MockOrch.return_value.retrieve.return_value = res
                with patch("app.retrieval.cli.get_settings", return_value=fake_settings):
                    rc = main(["x"])
    assert rc == 0
    call_kw = MockOrch.return_value.retrieve.call_args
    cfg = call_kw[0][1]
    assert cfg.mode == "dense_only"


def test_cli_manifest_not_found(fake_settings, capsys: pytest.CaptureFixture[str]) -> None:
    with patch("app.retrieval.cli.session_scope"):
        with patch(
            "app.retrieval.cli.resolve_manifest_record",
            side_effect=ManifestNotFoundError("none"),
        ):
            with patch("app.retrieval.cli.get_settings", return_value=fake_settings):
                rc = main(["q", "--json"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "indexing.cli" in err


def test_cli_metadata_filter_flags(fake_settings) -> None:
    res = RetrievalResultSet(query="q", mode="dense_only", results=[], manifest_hash="b" * 64)
    fake_manifest = MagicMock()
    fake_manifest.include_sparse = False

    with patch("app.retrieval.cli.session_scope"):
        with patch("app.retrieval.cli.resolve_manifest_record", return_value=fake_manifest):
            with patch("app.retrieval.cli.RetrievalOrchestrator") as MockOrch:
                MockOrch.return_value.retrieve.return_value = res
                with patch("app.retrieval.cli.get_settings", return_value=fake_settings):
                    rc = main(
                        [
                            "x",
                            "--mode",
                            "dense_only",
                            "--filter-jurisdiction",
                            "eu",
                            "--filter-legal-document-type",
                            "regulation",
                        ],
                    )
    assert rc == 0
    cfg = MockOrch.return_value.retrieve.call_args[0][1]
    assert cfg.metadata_filter == RetrievalMetadataFilter(
        jurisdiction="eu",
        legal_document_type="regulation",
    )


def test_cli_no_filter_flags(fake_settings) -> None:
    res = RetrievalResultSet(query="q", mode="dense_only", results=[], manifest_hash="b" * 64)
    fake_manifest = MagicMock()
    fake_manifest.include_sparse = False
    with patch("app.retrieval.cli.session_scope"):
        with patch("app.retrieval.cli.resolve_manifest_record", return_value=fake_manifest):
            with patch("app.retrieval.cli.RetrievalOrchestrator") as MockOrch:
                MockOrch.return_value.retrieve.return_value = res
                with patch("app.retrieval.cli.get_settings", return_value=fake_settings):
                    main(["x", "--mode", "dense_only"])
    cfg = MockOrch.return_value.retrieve.call_args[0][1]
    assert cfg.metadata_filter is None
