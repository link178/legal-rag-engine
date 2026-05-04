# API notes — Phases 8–9 (FastAPI v1 thin wrapper)

## Goal

Expose the **existing engine** over HTTP: ingestion, document listing, persisted chunk reads, pipeline chunk/index operator steps, manifest inspection, retrieval, and grounded answering. The API is a **thin layer** (routers + Pydantic + error mapping); it does **not** reimplement chunking, indexing, retrieval, or generation.

## Why `/v1/retrieve` and `/v1/answer` are separate

- **`POST /v1/retrieve`**: ranked chunks, scores (dense / sparse / RRF), manifest identity, and trace metadata—auditable retrieval without generation.
- **`POST /v1/answer`**: retrieval + `ContextBuilder` + `MockGenerationProvider` + mechanical `citation_verification`—end-to-end grounded output.

There is **no** `POST /v1/ask` (single combined endpoint remains out of scope).

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness (Phase 1). |
| `GET` | `/` | Service metadata. |
| `POST` | `/v1/ingest` | Ingest one local file path; optional Postgres persistence (`persist`, default `true`). |
| `GET` | `/v1/documents` | List persisted documents (`limit`, `offset`). |
| `GET` | `/v1/documents/{document_id}` | Document detail + `chunks_count`. |
| `GET` | `/v1/documents/{document_id}/chunks` | Paginated persisted chunks (`strategy`, `limit`, `offset`). Read-only DB. |
| `POST` | `/v1/chunk` | Run `chunk_document_persisted` for one document (operator step; persists `chunks` + `processing_runs`). |
| `POST` | `/v1/index` | Run `index_chunks_persisted` over the corpus manifest filter (operator step). |
| `GET` | `/v1/index-manifests` | List index manifests (`limit`, `offset`, optional filters). Read-only. |
| `GET` | `/v1/index-manifests/{manifest_id}` | Manifest detail. Read-only. |
| `POST` | `/v1/retrieve` | Hybrid / dense_only / sparse_only retrieval (read-only DB). |
| `POST` | `/v1/answer` | Grounded answer with **mock** provider only (read-only DB). |

## Pipeline endpoints (Phase 9)

The **minimal HTTP pipeline** is:

1. `POST /v1/ingest` (with `persist: true`)
2. `POST /v1/chunk` (per document; body mirrors `ChunkingConfig` defaults 1200/200/80 + `preserve_headings`)
3. `POST /v1/index` (optional filters; embedding family can follow **server** `Settings` like CLIs, or be overridden in the body)
4. `POST /v1/retrieve` / `POST /v1/answer` (pin `index_manifest_id` when you have multiple manifests)

Optional inspection:

- `GET /v1/index-manifests` / `GET /v1/index-manifests/{manifest_id}`
- `GET /v1/documents/{document_id}/chunks`

**CLI parity:** `python -m app.chunking.cli` and `python -m app.indexing.cli` remain supported and call the same runners as the HTTP routes.

## Prerequisites before `/v1/retrieve` or `/v1/answer`

Operators must have:

1. Migrated Postgres (`alembic upgrade head`).
2. Ingested at least one document (`POST /v1/ingest` with `persist: true`, or `python -m app.ingestion.cli … --persist`).
3. Chunked via **`POST /v1/chunk`** (or `python -m app.chunking.cli …`).
4. Indexed via **`POST /v1/index`** (or `python -m app.indexing.cli …`).

## Request contracts (high level)

- **Retrieval mode** must be engine literals: `dense_only`, `sparse_only`, `hybrid` (same as `app.retrieval.cli`).
- **Embedding family** for **retrieve/answer** (`EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`) is taken from **server** `Settings`, not from the request body—same as the CLIs.
- **`POST /v1/index`** accepts optional `embedding_provider`, `embedding_model`, `embedding_dimensions`; omitted fields default from `Settings` (same as `app.indexing.cli`).
- Optional **manifest pin**: `index_manifest_id` (UUID) in JSON body for retrieve/answer.
- **`/v1/answer`**: `provider` must be `"mock"`; validated **before** a DB session is opened.

## Errors

Structured JSON:

```json
{
  "error": {
    "code": "manifest_not_found",
    "message": "...",
    "details": {}
  }
}
```

See `app/api/errors.py` for the full mapping. Stack traces are never returned.

**Phase 9 codes (additions):** `chunking_failed` (422), `chunking_config_conflict` (409), `indexing_failed` (422), `no_chunks_to_index` (400). Pydantic `ValidationError` bodies (including model cross-field checks) return **422** `validation_error` with JSON-serializable `details.errors`.

## Explicit non-goals (Phases 8–9)

- No `POST /v1/ask`, multipart upload, or remote URL ingestion.
- No OpenAI / Anthropic / Ollama, streaming, or API keys.
- No auth, multi-tenant, billing, Redis, Qdrant, or background jobs.
- No new `processing_runs` rows for retrieval, answer, or **GET** manifest/document-chunk routes (read paths only). Ingest/chunk/index still record operator `processing_runs` when executed.

## curl examples

```bash
# Ingest (persist)
curl -s -X POST http://127.0.0.1:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"data/sample_corpus/basic/intro.md\",\"persist\":true}"

# Chunk (use document_id from ingest response)
curl -s -X POST http://127.0.0.1:8000/v1/chunk \
  -H "Content-Type: application/json" \
  -d "{\"document_id\":\"<document_id>\",\"strategy\":\"fixed_size\",\"chunk_size\":1200,\"chunk_overlap\":200}"

# Index (defaults for embedding family from server Settings if omitted)
curl -s -X POST http://127.0.0.1:8000/v1/index \
  -H "Content-Type: application/json" \
  -d "{\"chunking_strategy\":\"fixed_size\",\"include_dense\":true,\"include_sparse\":true}"

# List manifests
curl -s "http://127.0.0.1:8000/v1/index-manifests?limit=50&chunking_strategy=fixed_size"

# Manifest detail
curl -s "http://127.0.0.1:8000/v1/index-manifests/<manifest_id>"

# Document chunks (optional)
curl -s "http://127.0.0.1:8000/v1/documents/<document_id>/chunks?strategy=fixed_size&limit=100"

# List documents
curl -s "http://127.0.0.1:8000/v1/documents?limit=50&offset=0"

# Retrieve (pin manifest when multiple exist)
curl -s -X POST http://127.0.0.1:8000/v1/retrieve \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What does the intro describe?\",\"mode\":\"hybrid\",\"chunking_strategy\":\"fixed_size\",\"top_k\":5,\"index_manifest_id\":\"<manifest_id>\"}"

# Answer (mock only)
curl -s -X POST http://127.0.0.1:8000/v1/answer \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What does the intro describe?\",\"mode\":\"hybrid\",\"chunking_strategy\":\"fixed_size\",\"top_k\":5,\"provider\":\"mock\",\"index_manifest_id\":\"<manifest_id>\"}"
```

## Tests

- Default `pytest`: API unit tests under `tests/unit/test_api_*.py` (no Postgres).
- Optional DB smoke: `LEGAL_RAG_RUN_INTEGRATION_DB=1` and `pytest -m integration tests/integration/test_api_v1_persist.py`.
