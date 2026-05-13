"""POST /v1/chunk route tests (DB-free)."""

from __future__ import annotations

import contextlib
from uuid import uuid4

import pytest
from app.chunking.runner import ChunkingRunResult
from app.domain.models import Chunk, ProcessingRun
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("APP_NAME", "legal-rag-engine")
    from app.core.config import get_settings

    get_settings.cache_clear()
    return TestClient(app)


def test_chunk_overlap_validation(client: TestClient) -> None:
    r = client.post(
        "/v1/chunk",
        json={
            "document_id": str(uuid4()),
            "strategy": "fixed_size",
            "chunk_size": 100,
            "chunk_overlap": 100,
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_chunk_document_not_found_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    did = uuid4()
    result = ChunkingRunResult(
        document_id=did,
        run=None,
        strategy="fixed_size",
        created_chunks=0,
        skipped_existing=False,
        chunk_ids=[],
        error="Document not found",
    )
    monkeypatch.setattr(
        "app.api.routes.chunk.chunk_document_persisted",
        lambda *args, **kwargs: result,
    )

    class _Ctx:
        def __enter__(self):
            raise AssertionError("session_scope should not run")

        def __exit__(self, *a):  # noqa: ANN001
            return None

    monkeypatch.setattr(
        "app.api.routes.chunk.session_scope",
        lambda database_url=None: _Ctx(),
    )

    r = client.post("/v1/chunk", json={"document_id": str(did)})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "document_not_found"


def test_chunk_config_conflict_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    did = uuid4()
    msg = (
        "Existing chunks for this document and strategy were produced with a "
        "different chunking configuration (chunking_config_hash mismatch)."
    )
    result = ChunkingRunResult(
        document_id=did,
        run=None,
        strategy="fixed_size",
        created_chunks=0,
        skipped_existing=False,
        chunk_ids=[],
        error=msg,
    )
    monkeypatch.setattr(
        "app.api.routes.chunk.chunk_document_persisted",
        lambda *args, **kwargs: result,
    )

    monkeypatch.setattr(
        "app.api.routes.chunk.session_scope",
        lambda database_url=None: contextlib.nullcontext(None),
    )

    r = client.post("/v1/chunk", json={"document_id": str(did)})
    assert r.status_code == 409
    body = r.json()
    assert body["error"]["code"] == "chunking_config_conflict"
    assert msg in body["error"]["details"]["error"]


def test_chunk_generic_failure_422(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    did = uuid4()
    result = ChunkingRunResult(
        document_id=did,
        run=None,
        strategy="fixed_size",
        created_chunks=0,
        skipped_existing=False,
        chunk_ids=[],
        error="SomethingBad: rolled back",
    )
    monkeypatch.setattr(
        "app.api.routes.chunk.chunk_document_persisted",
        lambda *args, **kwargs: result,
    )
    monkeypatch.setattr(
        "app.api.routes.chunk.session_scope",
        lambda database_url=None: contextlib.nullcontext(None),
    )

    r = client.post("/v1/chunk", json={"document_id": str(did)})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "chunking_failed"
    assert r.json()["error"]["details"]["error"] == "SomethingBad: rolled back"


def test_chunk_happy_summary(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    did, rid, cid1, cid2 = uuid4(), uuid4(), uuid4(), uuid4()
    run = ProcessingRun(run_type="chunking", status="completed", id=rid)
    result = ChunkingRunResult(
        document_id=did,
        run=run,
        strategy="fixed_size",
        created_chunks=2,
        skipped_existing=False,
        chunk_ids=[cid1, cid2],
        error=None,
    )
    monkeypatch.setattr(
        "app.api.routes.chunk.chunk_document_persisted",
        lambda *args, **kwargs: result,
    )

    class _Repo:
        def __init__(self, _session):  # noqa: ANN001
            pass

        def count_by_document_and_strategy(self, doc_id, strategy):
            assert doc_id == did
            assert strategy == "fixed_size"
            return 2

        def list_by_document_and_strategy(self, doc_id, strategy):
            raise AssertionError("not called when include_chunks=false")

    monkeypatch.setattr("app.api.routes.chunk.ChunkRepository", _Repo)

    class _Ctx:
        def __enter__(self):  # noqa: ANN003
            return object()

        def __exit__(self, *_a):  # noqa: ANN003
            return None

    monkeypatch.setattr("app.api.routes.chunk.session_scope", lambda database_url=None: _Ctx())

    r = client.post("/v1/chunk", json={"document_id": str(did)})
    assert r.status_code == 200
    body = r.json()
    assert body["document_id"] == str(did)
    assert body["processing_run_id"] == str(rid)
    assert body["created"] is True
    assert body["skipped_existing"] is False
    assert body["chunks_count"] == 2
    assert len(body["chunk_ids"]) == 2
    assert body["chunks"] == []
    assert "config_hash" in body


def test_chunk_include_chunks_preview(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    did, rid = uuid4(), uuid4()
    cid = uuid4()
    run = ProcessingRun(run_type="chunking", status="completed", id=rid)
    result = ChunkingRunResult(
        document_id=did,
        run=run,
        strategy="fixed_size",
        created_chunks=1,
        skipped_existing=False,
        chunk_ids=[cid],
        error=None,
    )
    monkeypatch.setattr(
        "app.api.routes.chunk.chunk_document_persisted",
        lambda *args, **kwargs: result,
    )

    long_txt = "a" * 300
    ch = Chunk(
        id=cid,
        document_id=did,
        chunk_index=0,
        text=long_txt,
        chunking_strategy="fixed_size",
        char_count=300,
        checksum=None,
        metadata={"k": 1},
    )

    class _Repo:
        def __init__(self, _session):  # noqa: ANN001
            pass

        def count_by_document_and_strategy(self, _d, _s):
            return 1

        def list_by_document_and_strategy(self, _d, _s):
            return [ch]

    monkeypatch.setattr("app.api.routes.chunk.ChunkRepository", _Repo)

    class _Ctx:
        def __enter__(self):  # noqa: ANN003
            return object()

        def __exit__(self, *_a):  # noqa: ANN003
            return None

    monkeypatch.setattr("app.api.routes.chunk.session_scope", lambda database_url=None: _Ctx())

    r = client.post(
        "/v1/chunk",
        json={"document_id": str(did), "include_chunks": True, "chunk_preview_chars": 50},
    )
    assert r.status_code == 200
    chunks = r.json()["chunks"]
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == str(cid)
    assert len(chunks[0]["text_preview"]) <= 50


def test_chunk_idempotent_skip(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    did, rid = uuid4(), uuid4()
    run = ProcessingRun(run_type="chunking", status="completed", id=rid)
    cid = uuid4()
    result = ChunkingRunResult(
        document_id=did,
        run=run,
        strategy="fixed_size",
        created_chunks=0,
        skipped_existing=True,
        chunk_ids=[cid],
        error=None,
    )
    monkeypatch.setattr(
        "app.api.routes.chunk.chunk_document_persisted",
        lambda *args, **kwargs: result,
    )

    class _Repo:
        def __init__(self, _session):  # noqa: ANN001
            pass

        def count_by_document_and_strategy(self, _d, _s):
            return 1

        def list_by_document_and_strategy(self, _d, _s):
            return []

    monkeypatch.setattr("app.api.routes.chunk.ChunkRepository", _Repo)

    class _Ctx:
        def __enter__(self):  # noqa: ANN003
            return object()

        def __exit__(self, *_a):  # noqa: ANN003
            return None

    monkeypatch.setattr("app.api.routes.chunk.session_scope", lambda database_url=None: _Ctx())

    r = client.post("/v1/chunk", json={"document_id": str(did)})
    assert r.status_code == 200
    body = r.json()
    assert body["created"] is False
    assert body["skipped_existing"] is True
