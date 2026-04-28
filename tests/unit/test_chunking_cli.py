"""Unit tests for chunking CLI wiring (no DB)."""

from __future__ import annotations

import json
import uuid
from unittest.mock import patch

import pytest
from app.chunking.cli import main
from app.chunking.runner import ChunkingRunResult
from app.domain.models import ProcessingRun


@pytest.fixture
def fake_settings():
    class _S:
        database_url = "postgresql://localhost/test"

    return _S()


def test_cli_json_success_exit_code(fake_settings) -> None:
    doc_id = uuid.uuid4()
    run = ProcessingRun(run_type="chunking", status="completed", id=uuid.uuid4())
    fake = ChunkingRunResult(
        document_id=doc_id,
        run=run,
        strategy="fixed_size",
        created_chunks=1,
        skipped_existing=False,
        chunk_ids=[uuid.uuid4()],
        error=None,
    )
    with patch("app.chunking.cli.chunk_document_persisted", return_value=fake):
        with patch("app.chunking.cli.get_settings", return_value=fake_settings):
            rc = main([str(doc_id), "--strategy", "fixed_size", "--json"])
    assert rc == 0


def test_cli_invalid_uuid() -> None:
    rc = main(["not-a-uuid", "--strategy", "fixed_size"])
    assert rc == 1


def test_cli_json_payload_shape(fake_settings, capsys: pytest.CaptureFixture[str]) -> None:
    doc_id = uuid.uuid4()
    run = ProcessingRun(run_type="chunking", status="completed", id=uuid.uuid4())
    fake = ChunkingRunResult(
        document_id=doc_id,
        run=run,
        strategy="fixed_size",
        created_chunks=0,
        skipped_existing=True,
        chunk_ids=[],
        error=None,
    )
    with patch("app.chunking.cli.chunk_document_persisted", return_value=fake):
        with patch("app.chunking.cli.get_settings", return_value=fake_settings):
            rc = main([str(doc_id), "--strategy", "fixed_size", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["skipped_existing"] is True
    assert payload["strategy"] == "fixed_size"
