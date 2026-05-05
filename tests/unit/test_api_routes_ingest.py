"""POST /v1/ingest route tests (DB-free)."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from app.domain.models import Document, ProcessingRun
from app.ingestion.runner import IngestionRunResult
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("APP_NAME", "legal-rag-engine")
    from app.core.config import get_settings

    get_settings.cache_clear()
    return TestClient(app)


def test_ingest_persist_false_html_happy(client: TestClient, tmp_path: Path) -> None:
    p = tmp_path / "ing.html"
    p.write_text(
        "<html><head><title>API HTML</title></head><body><p>via api</p></body></html>",
        encoding="utf-8",
    )
    r = client.post("/v1/ingest", json={"path": str(p), "persist": False})
    assert r.status_code == 200
    body = r.json()
    assert body["source_type"] == "html"
    assert body["metadata"].get("format") == "html"
    assert body["checksum"]
    assert body["title"] == "API HTML"


def test_ingest_missing_path_404(client: TestClient) -> None:
    r = client.post("/v1/ingest", json={"path": str(Path("/nonexistent/file_xyz.md"))})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "document_not_found"


def test_ingest_unsupported_extension_415(client: TestClient, tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"\x00")
    r = client.post("/v1/ingest", json={"path": str(p), "persist": False})
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_document_type"


def test_ingest_persist_false_happy(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    tmp_path: Path,
) -> None:
    p = tmp_path / "a.md"
    p.write_text("# Hi", encoding="utf-8")

    doc = Document(
        source_path=str(p.resolve()),
        source_type="markdown",
        checksum="abc",
        title="T",
        metadata={"k": 1},
    )

    def _fake_service():
        class _Svc:
            def ingest_file(self, path: Path):
                assert path == p
                return doc

        return _Svc()

    monkeypatch.setattr("app.api.routes.ingest.default_ingestion_service", _fake_service)

    r = client.post("/v1/ingest", json={"path": str(p), "persist": False})
    assert r.status_code == 200
    body = r.json()
    assert body["persisted"] is False
    assert body["document_id"] is None
    assert body["processing_run_id"] is None
    assert body["checksum"] == "abc"
    assert body["created"] is None


def test_ingest_persist_true_happy(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    tmp_path: Path,
) -> None:
    p = tmp_path / "b.md"
    p.write_text("# Doc", encoding="utf-8")
    did = uuid4()
    rid = uuid4()
    d = Document(
        id=did,
        source_path=str(p),
        source_type="markdown",
        checksum="chk",
        title="X",
        metadata={},
    )
    run = ProcessingRun(run_type="ingest", status="completed", id=rid)

    def _fake_persist(path, service, database_url=None):
        return IngestionRunResult(document=d, run=run, created=True, skipped=False, error=None)

    monkeypatch.setattr("app.api.routes.ingest.ingest_file_persisted", _fake_persist)

    r = client.post("/v1/ingest", json={"path": str(p), "persist": True})
    assert r.status_code == 200
    body = r.json()
    assert body["persisted"] is True
    assert body["document_id"] == str(did)
    assert body["processing_run_id"] == str(rid)
    assert body["created"] is True


def test_ingest_persist_true_runner_error_422(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    tmp_path: Path,
) -> None:
    p = tmp_path / "c.md"
    p.write_text("# C", encoding="utf-8")

    def _fake_persist(path, service, database_url=None):
        return IngestionRunResult(
            document=None,
            run=None,
            created=False,
            skipped=False,
            error="OperationalError: connection refused",
        )

    monkeypatch.setattr("app.api.routes.ingest.ingest_file_persisted", _fake_persist)

    r = client.post("/v1/ingest", json={"path": str(p), "persist": True})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "ingestion_failed"
    assert "connection refused" in r.json()["error"]["details"]["error"]
