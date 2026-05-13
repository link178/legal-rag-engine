"""POST /v1/index and GET /v1/index-manifests route tests (DB-free)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from app.api.dependencies.db import get_session
from app.domain.models import ProcessingRun
from app.indexing.models import IndexedChunkResult, IndexingRunResult, IndexManifest
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("APP_NAME", "legal-rag-engine")
    from app.core.config import get_settings

    get_settings.cache_clear()
    return TestClient(app)


def test_index_both_paths_off_422(client: TestClient) -> None:
    r = client.post(
        "/v1/index",
        json={"include_dense": False, "include_sparse": False},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_index_embedding_dimensions_zero_422(client: TestClient) -> None:
    r = client.post("/v1/index", json={"embedding_dimensions": 0})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_index_no_chunks_400(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    err = "No chunks match the indexing filter (ingest/chunk rows first)"
    monkeypatch.setattr(
        "app.api.routes.index.index_chunks_persisted",
        lambda *a, **k: IndexingRunResult(manifest=None, skipped_existing=False, error=err),
    )
    r = client.post("/v1/index", json={})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "no_chunks_to_index"


def test_index_generic_failure_422(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.index.index_chunks_persisted",
        lambda *a, **k: IndexingRunResult(
            manifest=None,
            skipped_existing=False,
            error="OperationalError: boom",
        ),
    )
    r = client.post("/v1/index", json={})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "indexing_failed"
    assert r.json()["error"]["details"]["error"] == "OperationalError: boom"


def test_index_happy(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    mid, rid = uuid4(), uuid4()
    now = datetime.now(UTC)
    m = IndexManifest(
        id=mid,
        run_id=rid,
        corpus_version=None,
        embedding_provider="deterministic_hash",
        embedding_dimensions=16,
        include_sparse=True,
        include_dense=True,
        config_hash="c" * 64,
        chunk_set_hash="s" * 64,
        manifest_hash="m" * 64,
        chunking_strategy="fixed_size",
        document_count=1,
        chunk_count=3,
        indexed_chunk_count=3,
        embedding_model=None,
        embeddings_persisted=True,
        metadata={},
        created_at=now,
        updated_at=now,
    )
    run = ProcessingRun(run_type="indexing", status="completed", id=rid, metadata={"k": 1})
    res = IndexingRunResult(
        manifest=m,
        run=run,
        skipped_existing=False,
        chunk_results=[
            IndexedChunkResult(chunk_id=uuid4(), dense_indexed=True, sparse_indexed=True),
        ],
        error=None,
    )
    monkeypatch.setattr(
        "app.api.routes.index.index_chunks_persisted",
        lambda *a, **k: res,
    )

    r = client.post("/v1/index", json={"chunking_strategy": "fixed_size"})
    assert r.status_code == 200
    body = r.json()
    assert body["manifest_id"] == str(mid)
    assert body["created"] is True
    assert body["skipped_existing"] is False
    assert body["indexed_chunk_count"] == 3


def test_index_skip_idempotent(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    mid = uuid4()
    m = IndexManifest(
        id=mid,
        run_id=None,
        corpus_version=None,
        embedding_provider="deterministic_hash",
        embedding_dimensions=16,
        include_sparse=True,
        include_dense=True,
        config_hash="c" * 64,
        chunk_set_hash="s" * 64,
        manifest_hash="m" * 64,
        chunking_strategy=None,
        document_count=2,
        chunk_count=5,
        indexed_chunk_count=5,
        embedding_model=None,
        embeddings_persisted=True,
    )
    run = ProcessingRun(run_type="indexing", status="completed", id=uuid4())
    res = IndexingRunResult(
        manifest=m,
        run=run,
        skipped_existing=True,
        chunk_results=[],
        error=None,
    )
    monkeypatch.setattr(
        "app.api.routes.index.index_chunks_persisted",
        lambda *a, **k: res,
    )
    r = client.post("/v1/index", json={})
    assert r.status_code == 200
    assert r.json()["created"] is False
    assert r.json()["skipped_existing"] is True


def test_manifest_list_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    mid = uuid4()
    now = datetime.now(UTC)

    class Row:
        pass

    row = Row()
    row.id = mid
    row.manifest_hash = "a" * 64
    row.config_hash = "b" * 64
    row.chunk_set_hash = "c" * 64
    row.chunking_strategy = "fixed_size"
    row.embedding_provider = "deterministic_hash"
    row.embedding_model = None
    row.embedding_dimensions = 16
    row.include_dense = True
    row.include_sparse = True
    row.embeddings_persisted = True
    row.document_count = 1
    row.chunk_count = 2
    row.indexed_chunk_count = 2
    row.corpus_version = None
    row.created_at = now
    row.updated_at = now
    row.metadata_json = {"p": "4b"}

    calls: list = []

    class Repo:
        def __init__(self, session) -> None:  # noqa: ANN001
            calls.append(session)

        def list_recent(self, **kwargs):  # noqa: ANN003
            calls.append(kwargs)
            return [row]

    monkeypatch.setattr("app.api.routes.index.IndexManifestRepository", Repo)

    class Sess:
        pass

    def fake_session():
        yield Sess()

    app.dependency_overrides[get_session] = fake_session
    try:
        r = client.get(
            "/v1/index-manifests",
            params={"limit": 10, "chunking_strategy": "fixed_size", "include_dense": True},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert data["limit"] == 10
        assert data["manifests"][0]["manifest_id"] == str(mid)
        list_kw = next(x for x in calls if isinstance(x, dict))
        assert list_kw["chunking_strategy"] == "fixed_size"
        assert list_kw["include_dense"] is True
    finally:
        app.dependency_overrides.clear()


def test_manifest_detail_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class Repo:
        def __init__(self, session) -> None:  # noqa: ANN001
            pass

        def get_by_id(self, _mid):  # noqa: ANN001
            return None

    monkeypatch.setattr("app.api.routes.index.IndexManifestRepository", Repo)

    class Sess:
        pass

    def fake_session():
        yield Sess()

    app.dependency_overrides[get_session] = fake_session
    try:
        r = client.get(f"/v1/index-manifests/{uuid4()}")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "manifest_not_found"
    finally:
        app.dependency_overrides.clear()


def test_manifest_detail_200(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    mid = uuid4()
    now = datetime.now(UTC)

    class Row:
        pass

    row = Row()
    row.id = mid
    row.manifest_hash = "a" * 64
    row.config_hash = "b" * 64
    row.chunk_set_hash = "c" * 64
    row.chunking_strategy = "fixed_size"
    row.embedding_provider = "deterministic_hash"
    row.embedding_model = None
    row.embedding_dimensions = 16
    row.include_dense = True
    row.include_sparse = True
    row.embeddings_persisted = True
    row.document_count = 1
    row.chunk_count = 2
    row.indexed_chunk_count = 2
    row.corpus_version = None
    row.created_at = now
    row.updated_at = now
    row.metadata_json = {}

    class Repo:
        def __init__(self, session) -> None:  # noqa: ANN001
            pass

        def get_by_id(self, uid):
            return row if uid == mid else None

    monkeypatch.setattr("app.api.routes.index.IndexManifestRepository", Repo)

    class Sess:
        pass

    def fake_session():
        yield Sess()

    app.dependency_overrides[get_session] = fake_session
    try:
        r = client.get(f"/v1/index-manifests/{mid}")
        assert r.status_code == 200
        assert r.json()["manifest_id"] == str(mid)
    finally:
        app.dependency_overrides.clear()
