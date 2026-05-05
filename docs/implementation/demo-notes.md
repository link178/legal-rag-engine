# Demo notes — Phase 10 (Streamlit over `/v1`)

## Objective

Provide a **minimal, local-first UI** that exercises the same HTTP pipeline as the curl examples in [`README.md`](../../README.md): ingest → chunk → index → retrieve → answer (mock generation + citation verification). The demo **only** calls the public API; it does not import repositories, runners, or SQLAlchemy.

## Install

From the repository root (optional dev tools + demo):

```bash
pip install -e ".[dev,demo]"
```

Demo-only (Streamlit + httpx; no pytest/ruff/mypy):

```bash
pip install -e ".[demo]"
```

## Run order

1. **Postgres + pgvector** (matches default `DATABASE_URL` in `.env.example`):

   ```bash
   docker compose up -d postgres
   ```

2. **Migrations**:

   ```bash
   alembic upgrade head
   ```

3. **API**:

   ```bash
   python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

4. **Streamlit**:

   ```bash
   streamlit run app/ui/streamlit_app.py
   ```

Optional: copy `.env.example` → `.env` and adjust `DATABASE_URL` if your Postgres differs.

## Metadata filter (Phase 14)

The **Retrieval** and **Answer** tabs include an optional expander **“Metadata filter (optional)”** with seven text fields (`corpus_name`, `jurisdiction`, etc.). Only non-empty fields are sent as `metadata_filter` on `POST /v1/retrieve` and `POST /v1/answer` (omitted when all blank).

## Endpoints consumed

| UI action | Method | Path |
|-----------|--------|------|
| Health | `GET` | `/health` |
| Ingest | `POST` | `/v1/ingest` |
| List / use documents | `GET` | `/v1/documents` |
| Chunk | `POST` | `/v1/chunk` (`include_chunks=true` in this demo) |
| List chunks | `GET` | `/v1/documents/{document_id}/chunks` |
| Index | `POST` | `/v1/index` |
| List / pin manifests | `GET` | `/v1/index-manifests` |
| Retrieve | `POST` | `/v1/retrieve` |
| Answer | `POST` | `/v1/answer` (`provider` must be `mock`) |

## Recommended flow

1. **Check health** in the sidebar.
2. **Documents** — ingest `data/sample_corpus/basic/intro.md` (or **`.html` / `.pdf`** on the API host filesystem—same `POST /v1/ingest` path field) with `persist=true`, or pick an existing row and pin `document_id`.
3. **Chunking** — `fixed_size` (defaults 1200 / 200) → confirm `chunks_count` and previews.
4. **Indexing** — same `chunking_strategy` as chunking (`fixed_size`); leave embedding overrides empty to follow server `Settings` (CLI parity).
5. **Retrieval** — enable “Use pinned index_manifest_id” after indexing; use retrieval mode **`dense_only`**, **`sparse_only`**, or **`hybrid`** (not `dense` / `sparse`).
6. **Answer** — same retrieval mode / manifest pin; generation is **mock only**.

## Troubleshooting

### `manifest_not_found` (404)

- Complete ingest → chunk → index first.
- If you have **multiple** index manifests, pin the correct `manifest_id` in the sidebar (or use the **Pin** buttons after **Refresh manifest list**).
- **`hybrid`** retrieval requires a manifest built with **`include_sparse=true`**. If you indexed with sparse off, use `dense_only` or re-index with sparse enabled.

### `chunking_config_conflict` (409)

Chunks already exist for that document/strategy with a **different** config hash (e.g. different `chunk_size`). Change parameters to match existing rows, use another document, or adjust strategy/parameters consistently.

### `no_chunks_to_index` (400)

No chunk rows match the indexing filter (e.g. wrong `chunking_strategy`). Chunk the document first.

### `embedding_dimension_mismatch` (400)

Query vectors must match the manifest’s `embedding_dimensions`. Re-index with consistent `EMBEDDING_DIMENSIONS` / provider settings, or pin a manifest that matches server `Settings`.

### `unsupported_provider` (400) on `/v1/answer`

Only **`mock`** is supported in this phase. The UI keeps provider fixed to mock.

### Streamlit cannot reach API

- Confirm base URL in the sidebar (default `http://localhost:8000`).
- Firewalls / wrong host: use `127.0.0.1` if `localhost` fails.

## Tests

The HTTP client is covered by `tests/unit/test_demo_api_client.py` (`httpx.MockTransport`, no live API). The Streamlit layer is exercised manually.

## Limits (by design)

- No auth, uploads, PDF parsing, real LLMs, or background jobs.
- Ingest paths are **server-local** (API host filesystem), not browser uploads.
