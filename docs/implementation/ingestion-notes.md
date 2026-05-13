# Ingestion notes — Phase 2B (+ Phase 12 HTML/PDF)

## Goal

Provide the first real ingestion step: **local file → loader → conservative normalization → SHA-256 checksum of normalized text → domain `Document`**, with optional **operator-triggered** persistence to PostgreSQL and a **`processing_runs` trace**.

Loaders use **`Path.read_bytes()`** where needed so `Document.raw_text` matches decoding choices (HTML may use charset from `<meta>`; `.txt`/`.md` remain UTF-8 with explicit `DocumentLoadError` on invalid UTF-8).

## Supported formats

### Phase 2B (textual)

- `.txt` → `source_type` **`text`**
- `.md`, `.markdown` → `source_type` **`markdown`** (first `# Heading` as title when present; else filename stem)

### Phase 12 — HTML / PDF (local files only)

- `.html`, `.htm` → `source_type` **`html`**
  - Parsed with **BeautifulSoup** using the stdlib **`html.parser`** (no `lxml`).
  - Strips `<script>`, `<style>`, `<noscript>`, and HTML comments.
  - Visible text from block-level elements (headings, paragraphs, lists, simple table cells, etc.), blocks joined with blank lines before normalization.
  - **Title:** `<title>` text → first `<h1>` → filename stem.
  - **Metadata (JSON-safe):** `format: "html"`, optional `html_title`, optional `h1`.
  - **Errors:** empty extractable body → `DocumentLoadError` (HTTP **422** via API).

- `.pdf` → `source_type` **`pdf`**
  - Text extracted with **`pypdf`** (`strict=False`), **no OCR**. Per-page text joined with blank lines.
  - **Metadata:** `format: "pdf"`, `page_count` (always), optional `pdf_title` / `pdf_author` from document info when present (noise values like `untitled` / `anonymous` are omitted).
  - **Title:** PDF document title metadata when usable → else filename stem.
  - **Errors:** corrupt read → `DocumentLoadError`; encrypted PDF (password required) → `DocumentLoadError`; no extractable text (e.g. scanned pages) → `DocumentLoadError`.

**Explicitly out of scope:** OCR, remote URL fetch / crawling, perfect table or layout reconstruction, multipart upload.

**Phase 13:** local `legalize-*`-style corpus directories are handled by `app/ingestion/adapters/legal_corpus/` (operator CLI only); see [`legal-corpus-notes.md`](legal-corpus-notes.md).

## Checksums

- **`Document.checksum`**: SHA-256 **hex** of **normalized** text (UTF-8). Used with `source_path` for idempotent persistence (`uq_documents_source_path_checksum`).
- **`metadata["file_checksum_sha256"]`**: SHA-256 of **raw file bytes** (streaming read).

## Normalization

See `app/ingestion/normalizers/text_normalizer.py`: CRLF → LF, trim trailing spaces per line, collapse 3+ blank lines to 2, final strip. No aggressive cleanup.

## Errors

Controlled exceptions in `app/ingestion/errors.py` (subclasses of `IngestionError`): missing path, unsupported extension, load/decode failures.

## Pure vs persisted paths

- **`IngestionService.ingest_file`**: no SQLAlchemy; returns `Document` in memory. Optional `extra_metadata=` merges corpus-level fields into `Document.metadata` before `file_checksum_sha256` is set (Phase 13 adapter).
- **`ingest_file_persisted`** (`app/ingestion/runner.py`): one DB transaction per file; creates a `ProcessingRun`, inserts or skips `Document`, sets **`created_by_run_id`** on new rows, completes or fails the run.

## Operator CLI

```bash
python -m app.ingestion.cli path/to/file.md
python -m app.ingestion.cli path/to/file.html --json
python -m app.ingestion.cli path/to/file.pdf --json
python -m app.ingestion.cli path/to/file.md --persist
```

`--persist` requires a reachable Postgres and schema including `documents.created_by_run_id` (run `alembic upgrade head`).

**HTTP (Phase 8+):** `POST /v1/ingest` with `{"path":"<local-file>","persist":true}` uses the same `default_ingestion_service()` as this CLI (HTML/PDF included when the path exists on the API host).

## Tests

Default **`pytest`** stays **DB-free**. Opt-in integration: set `LEGAL_RAG_RUN_INTEGRATION_DB=1` and run `tests/integration/test_ingestion_persist.py` after migrations.

## Next phase candidates

- Dense/sparse indexing + pgvector columns (already in later phases)
