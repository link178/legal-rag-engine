"""Smoke tests for FastAPI bootstrap (no live Postgres required)."""

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
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "legal-rag-engine"
    assert body["environment"] == "staging"


def test_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "legal-rag-engine"
    assert body["status"] == "running"
    assert body["docs"] == "/docs"


def test_openapi_docs_available(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
