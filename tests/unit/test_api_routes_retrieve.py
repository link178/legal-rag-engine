"""POST /v1/retrieve route tests (DB-free)."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from app.api.dependencies.db import get_session
from app.main import app
from app.retrieval.errors import (
    EmbeddingDimensionMismatchError,
    ManifestNotFoundError,
    RetrieverNotConfiguredError,
)
from app.retrieval.models import RetrievalResultSet, RetrievedChunk
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("APP_NAME", "legal-rag-engine")
    from app.core.config import get_settings

    get_settings.cache_clear()
    return TestClient(app)


def test_retrieve_empty_query_422(client: TestClient) -> None:
    def _s():
        yield MagicMock()

    app.dependency_overrides[get_session] = _s
    try:
        r = client.post("/v1/retrieve", json={"query": ""})
        assert r.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_retrieve_mode_dense_rejected_422(client: TestClient) -> None:
    def _s():
        yield MagicMock()

    app.dependency_overrides[get_session] = _s
    try:
        r = client.post("/v1/retrieve", json={"query": "q", "mode": "dense"})
        assert r.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_retrieve_top_k_out_of_range_422(client: TestClient) -> None:
    def _s():
        yield MagicMock()

    app.dependency_overrides[get_session] = _s
    try:
        r = client.post("/v1/retrieve", json={"query": "q", "top_k": 0})
        assert r.status_code == 422
        r2 = client.post("/v1/retrieve", json={"query": "q", "top_k": 51})
        assert r2.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_retrieve_manifest_not_found_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_a, **_k):
        raise ManifestNotFoundError("no manifest")

    monkeypatch.setattr("app.api.routes.retrieve.resolve_manifest_record", _boom)

    class _Sess:
        pass

    def _fake_session():
        yield _Sess()

    app.dependency_overrides[get_session] = _fake_session
    try:
        r = client.post("/v1/retrieve", json={"query": "hello", "mode": "hybrid"})
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "manifest_not_found"
    finally:
        app.dependency_overrides.clear()


def test_retrieve_retriever_not_configured_400(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mf = MagicMock()
    mf.include_sparse = True
    monkeypatch.setattr("app.api.routes.retrieve.resolve_manifest_record", lambda *a, **k: mf)

    orch = MagicMock()
    orch.retrieve.side_effect = RetrieverNotConfiguredError("need both")

    def _mk(_d, _s):
        return orch

    monkeypatch.setattr("app.api.routes.retrieve.RetrievalOrchestrator", _mk)

    class _Sess:
        pass

    def _fake_session():
        yield _Sess()

    app.dependency_overrides[get_session] = _fake_session
    try:
        r = client.post("/v1/retrieve", json={"query": "x", "mode": "hybrid"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "retriever_not_configured"
    finally:
        app.dependency_overrides.clear()


def test_retrieve_embedding_mismatch_400(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mf = MagicMock()
    mf.include_sparse = True
    monkeypatch.setattr("app.api.routes.retrieve.resolve_manifest_record", lambda *a, **k: mf)

    orch = MagicMock()
    orch.retrieve.side_effect = EmbeddingDimensionMismatchError("bad dim")

    monkeypatch.setattr("app.api.routes.retrieve.RetrievalOrchestrator", lambda _d, _s: orch)

    class _Sess:
        pass

    def _fake_session():
        yield _Sess()

    app.dependency_overrides[get_session] = _fake_session
    try:
        r = client.post("/v1/retrieve", json={"query": "x", "mode": "dense_only"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "embedding_dimension_mismatch"
    finally:
        app.dependency_overrides.clear()


def test_retrieve_happy(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mf = MagicMock()
    mf.include_sparse = True
    monkeypatch.setattr("app.api.routes.retrieve.resolve_manifest_record", lambda *a, **k: mf)

    cid, did, mid = uuid4(), uuid4(), uuid4()
    chunk = RetrievedChunk(
        chunk_id=cid,
        document_id=did,
        text="x" * 300,
        source_path="p.md",
        title=None,
        heading=None,
        chunk_index=0,
        chunking_strategy="fixed_size",
        dense_score=0.5,
        sparse_score=None,
        rrf_score=0.02,
        rank_position=1,
        retrieval_sources=("dense",),
        metadata={"dense_distance": 1.0},
    )
    res = RetrievalResultSet(
        query="qq",
        mode="hybrid",
        results=[chunk],
        index_manifest_id=mid,
        manifest_hash="a" * 64,
        embedding_provider="deterministic_hash",
        embedding_model=None,
        embedding_dimensions=16,
        metadata={},
    )
    orch = MagicMock()
    orch.retrieve.return_value = res
    monkeypatch.setattr("app.api.routes.retrieve.RetrievalOrchestrator", lambda _d, _s: orch)

    class _Sess:
        pass

    def _fake_session():
        yield _Sess()

    app.dependency_overrides[get_session] = _fake_session
    try:
        r = client.post(
            "/v1/retrieve",
            json={"query": "qq", "mode": "hybrid", "top_k": 5},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "qq"
        assert body["mode"] == "hybrid"
        assert body["top_k"] == 5
        assert body["index_manifest_id"] == str(mid)
        assert body["manifest_hash"] == "a" * 64
        assert body["embedding_provider"] == "deterministic_hash"
        assert body["embedding_dimensions"] == 16
        assert body["total_results"] == 1
        assert body["chunks"][0]["chunk_id"] == str(cid)
        assert body["chunks"][0]["rank"] == 1
        assert body["chunks"][0]["dense_score"] == 0.5
        assert body["chunks"][0]["rrf_score"] == 0.02
        assert "text_preview" in body["chunks"][0]
    finally:
        app.dependency_overrides.clear()
