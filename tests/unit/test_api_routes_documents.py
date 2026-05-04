"""GET /v1/documents routes (DB-free via overrides)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from app.api.dependencies.db import get_session
from app.domain.models import Chunk, Document
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("APP_NAME", "legal-rag-engine")
    from app.core.config import get_settings

    get_settings.cache_clear()
    return TestClient(app)


def test_documents_list_limit_invalid(client: TestClient) -> None:
    """limit=0 triggers FastAPI Query validation → 422 envelope."""

    def _fake():
        yield None

    app.dependency_overrides[get_session] = _fake
    try:
        r = client.get("/v1/documents?limit=0")
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "validation_error"
    finally:
        app.dependency_overrides.clear()


def test_documents_list_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = uuid4()
    now = datetime.now(UTC)
    d = Document(
        id=doc_id,
        source_path="p.md",
        source_type="md",
        checksum="c",
        title="T",
        metadata={},
        created_at=now,
        updated_at=now,
    )

    class _Repo:
        def __init__(self, _session) -> None:
            pass

        def list_recent(self, limit: int, offset: int):
            assert limit == 50
            assert offset == 0
            return [d]

    monkeypatch.setattr("app.api.routes.documents.DocumentRepository", _Repo)

    class _Sess:
        pass

    def _fake_session():
        yield _Sess()

    app.dependency_overrides[get_session] = _fake_session
    try:
        r = client.get("/v1/documents")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 1
        assert body["documents"][0]["id"] == str(doc_id)
    finally:
        app.dependency_overrides.clear()


def test_documents_detail_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Repo:
        def __init__(self, _session) -> None:
            pass

        def get_by_id(self, _uid):
            return None

    monkeypatch.setattr("app.api.routes.documents.DocumentRepository", _Repo)

    class _Sess:
        pass

    def _fake_session():
        yield _Sess()

    app.dependency_overrides[get_session] = _fake_session
    try:
        r = client.get(f"/v1/documents/{uuid4()}")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "document_not_found"
    finally:
        app.dependency_overrides.clear()


def test_documents_detail_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = uuid4()
    now = datetime.now(UTC)
    d = Document(
        id=doc_id,
        source_path="p.md",
        source_type="md",
        checksum="c",
        title="T",
        metadata={},
        created_at=now,
        updated_at=now,
    )

    class _Repo:
        def __init__(self, _session) -> None:
            pass

        def get_by_id(self, uid: UUID):
            return d if uid == doc_id else None

    monkeypatch.setattr("app.api.routes.documents.DocumentRepository", _Repo)

    class _Sess:
        def scalar(self, _stmt):
            return 7

    def _fake_session():
        yield _Sess()

    app.dependency_overrides[get_session] = _fake_session
    try:
        r = client.get(f"/v1/documents/{doc_id}")
        assert r.status_code == 200
        assert r.json()["chunks_count"] == 7
    finally:
        app.dependency_overrides.clear()


def test_documents_chunks_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class _DocRepo:
        def __init__(self, _session) -> None:
            pass

        def get_by_id(self, _uid):
            return None

    monkeypatch.setattr("app.api.routes.documents.DocumentRepository", _DocRepo)

    class _Sess:
        pass

    def _fake_session():
        yield _Sess()

    app.dependency_overrides[get_session] = _fake_session
    try:
        r = client.get(f"/v1/documents/{uuid4()}/chunks")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "document_not_found"
    finally:
        app.dependency_overrides.clear()


def test_documents_chunks_with_strategy(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid4()
    chunk_id = uuid4()
    now = datetime.now(UTC)
    d = Document(
        id=doc_id,
        source_path="p.md",
        source_type="md",
        checksum="c",
        metadata={},
        created_at=now,
        updated_at=now,
    )
    ch = Chunk(
        id=chunk_id,
        document_id=doc_id,
        chunk_index=0,
        text="hello world slice",
        chunking_strategy="fixed_size",
        char_count=19,
        metadata={},
    )

    class _DocRepo:
        def __init__(self, _session) -> None:
            pass

        def get_by_id(self, uid: UUID):
            return d if uid == doc_id else None

    class _ChunkRepo:
        def __init__(self, _session) -> None:
            pass

        def count_by_document_and_strategy(self, uid, strat):
            assert uid == doc_id and strat == "fixed_size"
            return 10

        def list_by_document_and_strategy(self, uid, strat):
            assert uid == doc_id and strat == "fixed_size"
            return [ch]

    monkeypatch.setattr("app.api.routes.documents.DocumentRepository", _DocRepo)
    monkeypatch.setattr("app.api.routes.documents.ChunkRepository", _ChunkRepo)

    class _Sess:
        pass

    def _fake_session():
        yield _Sess()

    app.dependency_overrides[get_session] = _fake_session
    try:
        r = client.get(f"/v1/documents/{doc_id}/chunks?strategy=fixed_size&limit=1&offset=0")
        assert r.status_code == 200
        body = r.json()
        assert body["strategy"] == "fixed_size"
        assert body["count"] == 10
        assert body["limit"] == 1
        assert len(body["chunks"]) == 1
        assert body["chunks"][0]["chunk_id"] == str(chunk_id)
    finally:
        app.dependency_overrides.clear()


def test_documents_chunks_all_strategies(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = uuid4()
    now = datetime.now(UTC)
    d = Document(
        id=doc_id,
        source_path="p.md",
        source_type="md",
        checksum="c",
        metadata={},
        created_at=now,
        updated_at=now,
    )

    class _DocRepo:
        def __init__(self, _session) -> None:
            pass

        def get_by_id(self, uid: UUID):
            return d if uid == doc_id else None

    class _ChunkRepo:
        def __init__(self, _session) -> None:
            pass

        def list_by_document(self, uid):
            assert uid == doc_id
            return []

    monkeypatch.setattr("app.api.routes.documents.DocumentRepository", _DocRepo)
    monkeypatch.setattr("app.api.routes.documents.ChunkRepository", _ChunkRepo)

    class _Sess:
        def scalar(self, _stmt) -> int:
            return 0

    def _fake_session():
        yield _Sess()

    app.dependency_overrides[get_session] = _fake_session
    try:
        r = client.get(f"/v1/documents/{doc_id}/chunks")
        assert r.status_code == 200
        assert r.json()["count"] == 0
        assert r.json()["strategy"] is None
    finally:
        app.dependency_overrides.clear()
