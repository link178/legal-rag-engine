"""Unit tests for demo HTTP client (httpx.MockTransport; no Streamlit)."""

from __future__ import annotations

import json

import httpx
import pytest
from app.ui.api_client import ApiError, LegalRagApiClient


def _envelope(code: str, message: str, details: dict | None = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def test_health_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok", "service": "x", "environment": "dev"})

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        data = c.health()
    assert data["status"] == "ok"


def test_ingest_persisted() -> None:
    body = {
        "persisted": True,
        "document_id": "11111111-1111-1111-1111-111111111111",
        "processing_run_id": None,
        "source_path": "data/sample_corpus/basic/intro.md",
        "source_type": "markdown",
        "checksum": "abc",
        "title": "t",
        "metadata": {},
        "created": True,
        "skipped": False,
        "error": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/ingest"
        assert json.loads(request.content) == {
            "path": "data/sample_corpus/basic/intro.md",
            "persist": True,
        }
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        data = c.ingest("data/sample_corpus/basic/intro.md", persist=True)
    assert data["document_id"] == "11111111-1111-1111-1111-111111111111"
    assert data["checksum"] == "abc"


def test_chunk_include_chunks_in_request() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "document_id": captured["body"]["document_id"],
                "processing_run_id": None,
                "strategy": "fixed_size",
                "config_hash": "h",
                "created": True,
                "skipped_existing": False,
                "chunks_count": 1,
                "chunk_ids": [],
                "chunks": [{"chunk_id": "22222222-2222-2222-2222-222222222222"}],
                "metadata": {},
            },
        )

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        c.chunk(
            document_id="11111111-1111-1111-1111-111111111111",
            strategy="fixed_size",
            include_chunks=True,
        )
    assert captured["body"]["include_chunks"] is True


def test_index_skipped_existing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "manifest_id": "33333333-3333-3333-3333-333333333333",
                "manifest_hash": "mh",
                "config_hash": "ch",
                "chunk_set_hash": "csh",
                "chunking_strategy": "fixed_size",
                "embedding_provider": "deterministic_hash",
                "embedding_model": None,
                "embedding_dimensions": 16,
                "include_dense": True,
                "include_sparse": True,
                "embeddings_persisted": True,
                "chunk_count": 1,
                "indexed_chunk_count": 1,
                "failed_chunks_count": 0,
                "processing_run_id": None,
                "created": False,
                "skipped_existing": True,
                "metadata": {},
            },
        )

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        data = c.index(chunking_strategy="fixed_size")
    assert data["skipped_existing"] is True
    assert data["manifest_id"] == "33333333-3333-3333-3333-333333333333"


def test_unsupported_provider_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json=_envelope("unsupported_provider", "Only mock", {}),
        )

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        with pytest.raises(ApiError) as ei:
            c.answer(question="q?", provider="openai")
    err = ei.value
    assert err.status_code == 400
    assert err.code == "unsupported_provider"
    assert "mock" in err.message.lower() or "Only" in err.message


def test_manifest_not_found_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json=_envelope("manifest_not_found", "no index", {}))

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        with pytest.raises(ApiError) as ei:
            c.retrieve(query="q", mode="hybrid")
    assert ei.value.code == "manifest_not_found"


def test_chunking_config_conflict_409() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json=_envelope(
                "chunking_config_conflict",
                "Chunking configuration conflict.",
                {"error": "x"},
            ),
        )

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        with pytest.raises(ApiError) as ei:
            c.chunk(document_id="11111111-1111-1111-1111-111111111111")
    assert ei.value.status_code == 409
    assert ei.value.code == "chunking_config_conflict"


def test_connect_error_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope", request=request)

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        with pytest.raises(ApiError) as ei:
            c.health()
    assert ei.value.code == "transport_error"
    assert ei.value.status_code == 0


def test_timeout_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout", request=request)

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        with pytest.raises(ApiError) as ei:
            c.health()
    assert ei.value.code == "transport_error"


def test_non_json_502() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, content=b"bad gateway html")

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        with pytest.raises(ApiError) as ei:
            c.health()
    err = ei.value
    assert err.status_code == 502
    assert err.code in ("invalid_response", "http_error")


def test_validation_error_422() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json=_envelope("validation_error", "Request validation failed.", {"errors": []}),
        )

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        with pytest.raises(ApiError) as ei:
            c.retrieve(query="", mode="hybrid")
    assert ei.value.code == "validation_error"


def test_retrieve_sends_correct_mode_literal() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "query": "x",
                "mode": "dense_only",
                "top_k": 5,
                "total_results": 0,
                "chunks": [],
                "metadata": {},
            },
        )

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        c.retrieve(query="x", mode="dense_only")
    assert captured["body"]["mode"] == "dense_only"


def test_retrieve_includes_metadata_filter_when_set() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "query": "x",
                "mode": "hybrid",
                "top_k": 5,
                "total_results": 0,
                "chunks": [],
                "metadata": {},
            },
        )

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        c.retrieve(query="x", metadata_filter={"jurisdiction": "eu"})
    assert captured["body"]["metadata_filter"] == {"jurisdiction": "eu"}


def test_retrieve_omits_metadata_filter_when_none() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "query": "x",
                "mode": "hybrid",
                "top_k": 5,
                "total_results": 0,
                "chunks": [],
                "metadata": {},
            },
        )

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        c.retrieve(query="x")
    assert "metadata_filter" not in captured["body"]


def test_answer_includes_metadata_filter_when_set() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "question": "q",
                "answer": "a",
                "mode": "grounded",
                "insufficient_context": False,
                "citations": [],
                "used_citation_ids": [],
                "metadata": {},
            },
        )

    transport = httpx.MockTransport(handler)
    with LegalRagApiClient("http://test", transport=transport) as c:
        c.answer(question="q", metadata_filter={"jurisdiction": "eu"})
    assert captured["body"]["metadata_filter"] == {"jurisdiction": "eu"}
