"""API error envelope handlers."""

from __future__ import annotations

from app.api.errors import (
    ChunkConfigConflictError,
    ChunkingFailedError,
    IndexingFailedError,
    IngestionFailedError,
    NoChunksToIndexError,
    PersistedDocumentNotFoundError,
    register_exception_handlers,
)
from app.generation.errors import GenerationError, UnsupportedProviderError
from app.ingestion.errors import (
    DocumentLoadError,
    UnsupportedDocumentTypeError,
)
from app.ingestion.errors import (
    DocumentNotFoundError as IngestDocumentNotFoundError,
)
from app.retrieval.errors import (
    EmbeddingDimensionMismatchError,
    EmptyQueryError,
    ManifestNotFoundError,
    RetrieverNotConfiguredError,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client() -> TestClient:
    app = FastAPI()

    @app.get("/empty")
    async def _empty():
        raise EmptyQueryError("bad")

    @app.get("/manifest")
    async def _manifest():
        raise ManifestNotFoundError("none")

    @app.get("/ingest-path")
    async def _ingest_path():
        raise IngestDocumentNotFoundError("missing")

    @app.get("/persisted-doc")
    async def _persisted():
        raise PersistedDocumentNotFoundError()

    @app.get("/415")
    async def _415():
        raise UnsupportedDocumentTypeError("nope")

    @app.get("/load")
    async def _load():
        raise DocumentLoadError("read fail")

    @app.get("/provider")
    async def _provider():
        raise UnsupportedProviderError("only mock")

    @app.get("/gen")
    async def _gen():
        raise GenerationError("boom")

    @app.get("/retriever")
    async def _ret():
        raise RetrieverNotConfiguredError("need dense")

    @app.get("/dim")
    async def _dim():
        raise EmbeddingDimensionMismatchError("mismatch")

    @app.get("/value")
    async def _value():
        raise ValueError("invalid")

    @app.get("/ingest-fail")
    async def _ingest_fail():
        raise IngestionFailedError("db down")

    @app.get("/boom")
    async def _boom():
        raise RuntimeError("secret")

    @app.get("/chunk_fail")
    async def _chunk_fail():
        raise ChunkingFailedError("bad chunk")

    @app.get("/chunk_conflict")
    async def _chunk_conflict():
        raise ChunkConfigConflictError("hash clash")

    @app.get("/index_fail")
    async def _index_fail():
        raise IndexingFailedError("bad index")

    @app.get("/no_chunks")
    async def _no_chunks():
        raise NoChunksToIndexError("nothing to index")

    register_exception_handlers(app)
    return TestClient(app, raise_server_exceptions=False)


def _assert_envelope(res, status: int, code: str) -> None:
    assert res.status_code == status
    body = res.json()
    assert "error" in body
    assert body["error"]["code"] == code
    assert "message" in body["error"]


def test_empty_query_400() -> None:
    c = _client()
    r = c.get("/empty")
    _assert_envelope(r, 400, "empty_query")


def test_manifest_404() -> None:
    c = _client()
    r = c.get("/manifest")
    _assert_envelope(r, 404, "manifest_not_found")


def test_ingest_path_404() -> None:
    c = _client()
    r = c.get("/ingest-path")
    _assert_envelope(r, 404, "document_not_found")


def test_persisted_doc_404() -> None:
    c = _client()
    r = c.get("/persisted-doc")
    _assert_envelope(r, 404, "document_not_found")


def test_unsupported_type_415() -> None:
    c = _client()
    r = c.get("/415")
    _assert_envelope(r, 415, "unsupported_document_type")


def test_document_load_422() -> None:
    c = _client()
    r = c.get("/load")
    _assert_envelope(r, 422, "document_load_failed")


def test_unsupported_provider_400() -> None:
    c = _client()
    r = c.get("/provider")
    _assert_envelope(r, 400, "unsupported_provider")


def test_generation_error_500() -> None:
    c = _client()
    r = c.get("/gen")
    _assert_envelope(r, 500, "generation_error")


def test_retriever_400() -> None:
    c = _client()
    r = c.get("/retriever")
    _assert_envelope(r, 400, "retriever_not_configured")


def test_embedding_mismatch_400() -> None:
    c = _client()
    r = c.get("/dim")
    _assert_envelope(r, 400, "embedding_dimension_mismatch")


def test_value_error_400() -> None:
    c = _client()
    r = c.get("/value")
    _assert_envelope(r, 400, "invalid_config")


def test_ingestion_failed_422() -> None:
    c = _client()
    r = c.get("/ingest-fail")
    _assert_envelope(r, 422, "ingestion_failed")
    assert r.json()["error"]["details"].get("error") == "db down"


def test_unhandled_500_redacted() -> None:
    c = _client()
    r = c.get("/boom")
    _assert_envelope(r, 500, "internal_error")
    assert "secret" not in r.text


def test_chunking_failed_422() -> None:
    c = _client()
    r = c.get("/chunk_fail")
    _assert_envelope(r, 422, "chunking_failed")
    assert r.json()["error"]["details"]["error"] == "bad chunk"


def test_chunking_config_conflict_409() -> None:
    c = _client()
    r = c.get("/chunk_conflict")
    _assert_envelope(r, 409, "chunking_config_conflict")
    assert r.json()["error"]["details"]["error"] == "hash clash"


def test_indexing_failed_422() -> None:
    c = _client()
    r = c.get("/index_fail")
    _assert_envelope(r, 422, "indexing_failed")
    assert r.json()["error"]["details"]["error"] == "bad index"


def test_no_chunks_to_index_400() -> None:
    c = _client()
    r = c.get("/no_chunks")
    _assert_envelope(r, 400, "no_chunks_to_index")
    assert r.json()["error"]["details"]["error"] == "nothing to index"
