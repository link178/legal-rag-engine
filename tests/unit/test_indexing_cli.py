"""CLI wiring for persisted indexing (mocked runner; no DB)."""

from __future__ import annotations

import json
import uuid
from unittest.mock import patch
from uuid import uuid4

import pytest
from app.domain.models import ProcessingRun
from app.indexing.cli import main
from app.indexing.models import IndexingRunResult, IndexManifest


@pytest.fixture
def fake_settings():
    class _S:
        database_url = "postgresql://localhost/test"
        embedding_provider = "deterministic_hash"
        embedding_dimensions = 16
        embedding_model = ""

    return _S()


def test_cli_json_success(fake_settings, capsys: pytest.CaptureFixture[str]) -> None:
    mid = uuid4()
    run = ProcessingRun(run_type="indexing", status="completed", id=uuid.uuid4())
    m = IndexManifest(
        embedding_provider="deterministic_hash",
        embedding_dimensions=16,
        include_sparse=True,
        include_dense=True,
        config_hash="a",
        chunk_set_hash="b",
        manifest_hash="c" * 64,
        id=mid,
        embedding_model=None,
        embeddings_persisted=True,
    )
    fake = IndexingRunResult(
        manifest=m,
        run=run,
        skipped_existing=False,
        chunk_results=[],
    )
    with patch("app.indexing.cli.index_chunks_persisted", return_value=fake):
        with patch("app.indexing.cli.get_settings", return_value=fake_settings):
            rc = main(["--chunking-strategy", "fixed_size", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["skipped_existing"] is False
    assert payload["manifest_id"] == str(mid)
    assert payload["embedding_provider"] == "deterministic_hash"
    assert payload["embeddings_persisted"] is True
    assert payload["embedding_model"] is None


def test_cli_embedding_model_passed_to_runner(fake_settings) -> None:
    captured: list[object] = []

    def capture(cfg, **kwargs):
        captured.append(cfg)
        return IndexingRunResult(manifest=None, error="stop")

    with patch("app.indexing.cli.index_chunks_persisted", side_effect=capture):
        with patch("app.indexing.cli.get_settings", return_value=fake_settings):
            rc = main(
                [
                    "--chunking-strategy",
                    "fixed_size",
                    "--embedding-model",
                    "intfloat/multilingual-e5-small",
                    "--json",
                ]
            )
    assert rc == 1
    assert len(captured) == 1
    cfg = captured[0]
    assert cfg.embedding_model == "intfloat/multilingual-e5-small"


def test_cli_dense_sparse_mutex(fake_settings) -> None:
    rc = main(["--no-dense", "--no-sparse"])
    assert rc == 1


def test_cli_skipped_branch_payload(fake_settings, capsys: pytest.CaptureFixture[str]) -> None:
    mh = "d" * 64
    existing = IndexManifest(
        embedding_provider="deterministic_hash",
        embedding_dimensions=16,
        include_sparse=True,
        include_dense=True,
        config_hash="a",
        chunk_set_hash="b",
        manifest_hash=mh,
        id=uuid4(),
        embeddings_persisted=True,
    )
    run = ProcessingRun(run_type="indexing", status="completed", id=uuid.uuid4())
    fake = IndexingRunResult(
        manifest=existing,
        run=run,
        skipped_existing=True,
    )
    with patch("app.indexing.cli.index_chunks_persisted", return_value=fake):
        with patch("app.indexing.cli.get_settings", return_value=fake_settings):
            rc = main(["--chunking-strategy", "fixed_size", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["skipped_existing"] is True
