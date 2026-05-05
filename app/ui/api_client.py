"""HTTP client for the Legal RAG Engine demo (consumes FastAPI only)."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx

# --- Error model --------------------------------------------------------------------


class ApiError(Exception):
    """Structured API or transport failure."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any],
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        self.raw = raw
        super().__init__(f"[{status_code}] {code}: {message}")


def _parse_error_body(
    status_code: int,
    body_text: str | bytes | None,
) -> tuple[str, str, dict[str, Any], dict[str, Any] | None]:
    """Return (code, message, details, raw_dict or None)."""
    if body_text in (None, "", b""):
        return (
            "http_error",
            f"HTTP {status_code}",
            {},
            None,
        )
    if isinstance(body_text, bytes):
        text = body_text.decode("utf-8", errors="replace")
    elif isinstance(body_text, str):
        text = body_text
    else:
        text = ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return (
            "invalid_response",
            text[:500] if len(text) > 500 else text,
            {},
            None,
        )
    if isinstance(data, dict) and "error" in data and isinstance(data["error"], dict):
        err = data["error"]
        code = str(err.get("code", "error"))
        msg = str(err.get("message", ""))
        details = err.get("details")
        if not isinstance(details, dict):
            details = {}
        return code, msg, details, data
    return "invalid_response", "Unexpected JSON shape", {"body": text[:500]}, data


# --- Client -------------------------------------------------------------------------


class LegalRagApiClient:
    """Thin synchronous wrapper over the public HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        u = base_url.rstrip("/") or "http://localhost:8000"
        self._base = u
        self._timeout = httpx.Timeout(timeout_s)
        client_kw: dict[str, Any] = {"base_url": u, "timeout": self._timeout}
        if transport is not None:
            client_kw["transport"] = transport
        self._client = httpx.Client(**client_kw)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LegalRagApiClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            r = self._client.request(method, path, params=params, json=json_body)
        except httpx.ConnectError as e:
            raise ApiError(
                status_code=0,
                code="transport_error",
                message=f"Could not connect: {e}",
                details={},
            ) from e
        except httpx.TimeoutException as e:
            raise ApiError(
                status_code=0,
                code="transport_error",
                message=f"Request timed out: {e}",
                details={},
            ) from e
        except httpx.RequestError as e:
            raise ApiError(
                status_code=0,
                code="transport_error",
                message=str(e),
                details={},
            ) from e

        if 200 <= r.status_code < 300:
            try:
                out = r.json()
            except json.JSONDecodeError:
                raise ApiError(
                    status_code=r.status_code,
                    code="invalid_response",
                    message="Response body is not valid JSON",
                    details={"text_preview": r.text[:500]},
                )
            if not isinstance(out, dict):
                raise ApiError(
                    status_code=r.status_code,
                    code="invalid_response",
                    message="JSON root must be an object",
                    details={},
                )
            return out

        code, msg, details, raw = _parse_error_body(r.status_code, r.text)
        raise ApiError(
            status_code=r.status_code,
            code=code,
            message=msg,
            details=details,
            raw=raw,
        )

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def ingest(self, path: str, *, persist: bool = True) -> dict[str, Any]:
        return self._request("POST", "/v1/ingest", json_body={"path": path, "persist": persist})

    def list_documents(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        return self._request(
            "GET",
            "/v1/documents",
            params={"limit": limit, "offset": offset},
        )

    def get_document(self, document_id: str | UUID) -> dict[str, Any]:
        did = str(document_id)
        return self._request("GET", f"/v1/documents/{quote(did, safe='')}")

    def list_document_chunks(
        self,
        document_id: str | UUID,
        *,
        strategy: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        did = str(document_id)
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if strategy is not None and strategy.strip():
            params["strategy"] = strategy.strip()
        return self._request(
            "GET",
            f"/v1/documents/{quote(did, safe='')}/chunks",
            params=params,
        )

    def chunk(
        self,
        *,
        document_id: str | UUID,
        strategy: str = "fixed_size",
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_chunk_chars: int = 80,
        preserve_headings: bool = True,
        include_chunks: bool = True,
        chunk_preview_chars: int = 240,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "document_id": str(document_id),
            "strategy": strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "min_chunk_chars": min_chunk_chars,
            "preserve_headings": preserve_headings,
            "include_chunks": include_chunks,
            "chunk_preview_chars": chunk_preview_chars,
        }
        return self._request("POST", "/v1/chunk", json_body=body)

    def index(
        self,
        *,
        chunking_strategy: str | None = None,
        include_dense: bool = True,
        include_sparse: bool = True,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
        batch_size: int = 32,
        force_reindex: bool = False,
        corpus_version: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "include_dense": include_dense,
            "include_sparse": include_sparse,
            "batch_size": batch_size,
            "force_reindex": force_reindex,
        }
        if chunking_strategy is not None and str(chunking_strategy).strip():
            body["chunking_strategy"] = str(chunking_strategy).strip()
        if embedding_provider is not None and str(embedding_provider).strip():
            body["embedding_provider"] = str(embedding_provider).strip()
        if embedding_model is not None and str(embedding_model).strip():
            body["embedding_model"] = str(embedding_model).strip()
        if embedding_dimensions is not None:
            body["embedding_dimensions"] = embedding_dimensions
        if corpus_version is not None and str(corpus_version).strip():
            body["corpus_version"] = str(corpus_version).strip()
        return self._request("POST", "/v1/index", json_body=body)

    def list_manifests(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        chunking_strategy: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
        include_sparse: bool | None = None,
        include_dense: bool | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if chunking_strategy is not None and str(chunking_strategy).strip():
            params["chunking_strategy"] = str(chunking_strategy).strip()
        if embedding_provider is not None and str(embedding_provider).strip():
            params["embedding_provider"] = str(embedding_provider).strip()
        if embedding_model is not None and str(embedding_model).strip():
            params["embedding_model"] = str(embedding_model).strip()
        if embedding_dimensions is not None:
            params["embedding_dimensions"] = embedding_dimensions
        if include_sparse is not None:
            params["include_sparse"] = include_sparse
        if include_dense is not None:
            params["include_dense"] = include_dense
        return self._request("GET", "/v1/index-manifests", params=params)

    def get_manifest(self, manifest_id: str | UUID) -> dict[str, Any]:
        mid = str(manifest_id)
        return self._request("GET", f"/v1/index-manifests/{quote(mid, safe='')}")

    def retrieve(
        self,
        *,
        query: str,
        mode: str = "hybrid",
        chunking_strategy: str | None = None,
        top_k: int = 5,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        rrf_k: int = 60,
        index_manifest_id: str | UUID | None = None,
        metadata_filter: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "dense_top_k": dense_top_k,
            "sparse_top_k": sparse_top_k,
            "rrf_k": rrf_k,
        }
        if chunking_strategy is not None and str(chunking_strategy).strip():
            body["chunking_strategy"] = str(chunking_strategy).strip()
        if index_manifest_id is not None:
            body["index_manifest_id"] = str(index_manifest_id)
        if metadata_filter:
            body["metadata_filter"] = dict(metadata_filter)
        return self._request("POST", "/v1/retrieve", json_body=body)

    def answer(
        self,
        *,
        question: str,
        mode: str = "hybrid",
        chunking_strategy: str | None = None,
        top_k: int = 5,
        provider: str = "mock",
        index_manifest_id: str | UUID | None = None,
        max_chunks: int = 5,
        max_context_chars: int = 8000,
        max_chunk_chars: int = 2000,
        min_score: float | None = None,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        rrf_k: int = 60,
        metadata_filter: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "question": question,
            "mode": mode,
            "top_k": top_k,
            "provider": provider,
            "max_chunks": max_chunks,
            "max_context_chars": max_context_chars,
            "max_chunk_chars": max_chunk_chars,
            "dense_top_k": dense_top_k,
            "sparse_top_k": sparse_top_k,
            "rrf_k": rrf_k,
        }
        if chunking_strategy is not None and str(chunking_strategy).strip():
            body["chunking_strategy"] = str(chunking_strategy).strip()
        if index_manifest_id is not None:
            body["index_manifest_id"] = str(index_manifest_id)
        if min_score is not None:
            body["min_score"] = min_score
        if metadata_filter:
            body["metadata_filter"] = dict(metadata_filter)
        return self._request("POST", "/v1/answer", json_body=body)
