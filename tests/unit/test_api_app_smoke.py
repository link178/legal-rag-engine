"""Smoke: OpenAPI paths and health unchanged after Phase 8+9."""

from __future__ import annotations

import pytest
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("APP_NAME", "legal-rag-engine")
    monkeypatch.setenv("APP_ENV", "staging")
    from app.core.config import get_settings

    get_settings.cache_clear()
    return TestClient(app)


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root_ok(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["docs"] == "/docs"


def test_openapi_lists_v1_paths(client: TestClient) -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json().get("paths", {})
    assert "/v1/ingest" in paths
    assert "/v1/documents" in paths
    assert "/v1/retrieve" in paths
    assert "/v1/answer" in paths
    assert "/v1/documents/{document_id}" in paths
    assert "/v1/chunk" in paths
    assert "/v1/index" in paths
    assert "/v1/index-manifests" in paths
    assert "/v1/index-manifests/{manifest_id}" in paths
    assert "/v1/documents/{document_id}/chunks" in paths
