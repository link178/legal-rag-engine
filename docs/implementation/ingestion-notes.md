# Ingestion notes — Phase 2B

## Goal

Provide the first real ingestion step: **local file → loader → conservative normalization → SHA-256 checksum of normalized text → domain `Document`**, with optional **operator-triggered** persistence to PostgreSQL and a **`processing_runs` trace**.

Loaders use **`Path.read_bytes()`** then UTF-8 decode so `Document.raw_text` matches on-disk bytes (Windows text-mode reads would otherwise normalize newlines).

## Supported formats (this phase)

- `.txt` → `source_type` **`text`**
- `.md`, `.markdown` → `source_type` **`markdown`** (first `# Heading` as title when present; else filename stem)

Out of scope here: PDF, HTML, `legalize-*` adapters, chunking, embeddings, retrieval, generation, `/v1/ingest`, Streamlit.

## Checksums

- **`Document.checksum`**: SHA-256 **hex** of **normalized** text (UTF-8). Used with `source_path` for idempotent persistence (`uq_documents_source_path_checksum`).
- **`metadata["file_checksum_sha256"]`**: SHA-256 of **raw file bytes** (streaming read).

## Normalization

See `app/ingestion/normalizers/text_normalizer.py`: CRLF → LF, trim trailing spaces per line, collapse 3+ blank lines to 2, final strip. No aggressive cleanup.

## Errors

Controlled exceptions in `app/ingestion/errors.py` (subclasses of `IngestionError`): missing path, unsupported extension, load/decode failures.

## Pure vs persisted paths

- **`IngestionService.ingest_file`**: no SQLAlchemy; returns `Document` in memory.
- **`ingest_file_persisted`** (`app/ingestion/runner.py`): one DB transaction per file; creates a `ProcessingRun`, inserts or skips `Document`, sets **`created_by_run_id`** on new rows, completes or fails the run.

## Operator CLI

```bash
python -m app.ingestion.cli path/to/file.md
python -m app.ingestion.cli path/to/file.md --json
python -m app.ingestion.cli path/to/file.md --persist
```

`--persist` requires a reachable Postgres and schema including `documents.created_by_run_id` (run `alembic upgrade head`).

## Tests

Default **`pytest`** stays **DB-free**. Opt-in integration: set `LEGAL_RAG_RUN_INTEGRATION_DB=1` and run `tests/integration/test_ingestion_persist.py` after migrations.

## Next phase candidates

- PDF / HTML / `legalize-*` adapter
- Dense/sparse indexing + pgvector columns
- API `POST /v1/ingest` (engine-first: API calls the same services as the CLI)
