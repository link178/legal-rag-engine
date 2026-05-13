# Chunking notes — Phase 3

## Goal

Add the second real RAG pipeline layer: **persisted document row → deterministic chunking strategies → `Chunk` domain objects → optional persistence in PostgreSQL**, with operator-facing trace via `processing_runs` and per-chunk lineage.

This phase intentionally does **not** add embeddings, vector columns, Postgres FTS, retrieval, generation, `/v1/*` API routes, or Streamlit.

## Why chunking is a separate phase

Chunking policies strongly affect retrieval quality and evaluation comparability. Keeping chunking isolated:

- keeps strategies testable without Docker or Postgres (pure unit tests);
- keeps persisted runs auditable (`run_type="chunking"`);
- avoids coupling chunk boundaries to embedding models or index formats.

## Strategies implemented

### `fixed_size`

Character windows with overlap. Prefers splitting near the end of a window on:

1. `\n\n`
2. `\n`
3. space
4. hard cut

### `structure_aware`

Baseline Markdown-aware splitting:

- Detects ATX headings `#` … `######` at the start of a line.
- Subdivides oversized sections using the same fixed-size window logic.
- If there are **no headings** (or `--no-preserve-headings` / `preserve_headings=False`), it falls back to fixed-size behavior while still labeling the outer strategy as `structure_aware` and recording `metadata.fallback`.

Out of scope for this phase:

- legal article / statute-aware splitting;
- PDF page-aware splitting;
- semantic / embedding-based chunking;
- destructive re-chunking / automatic replacement policies.

## Configuration and traceability

`ChunkingConfig` is validated (`chunk_overlap < chunk_size`, etc.) and serialized to metadata on every chunk as:

- `metadata.chunking_config` (canonical JSON-serializable fields)
- `metadata.chunking_config_hash` (SHA-256 over canonical JSON)

Each chunk also includes:

- `metadata.start_offset` / `metadata.end_offset` (offsets into the **normalized/raw** text string used for chunking)
- `metadata.source_document_checksum` (the domain `Document.checksum`)
- `checksum` on the chunk body (SHA-256 of UTF-8 chunk text)

## Persistence model

- Chunks store `created_by_run_id → processing_runs.id` (`ON DELETE SET NULL`) for audit.
- Idempotency for persistence is keyed by **`(document_id, chunking_strategy)` + `chunking_config_hash`**:
  - If matching chunks already exist with the same hash: the run completes with `skipped=true` and **no duplicate rows**.
  - If chunks exist but the hash differs: the run fails with a clear error (**no deletes** in Phase 3).

## Operator entry points

Pure chunking (no DB):

- use `ChunkingService.chunk_document(...)` in code; `Document.id` must be set to build `Chunk` rows.

Persisted chunking:

- `python -m app.chunking.cli <document_uuid> --strategy fixed_size [--json]`
- `python -m app.chunking.cli <document_uuid> --strategy structure_aware [--json]`

Requires Postgres, `DATABASE_URL`, and `alembic upgrade head` (including `chunks.created_by_run_id`).

## Tests

- Default `pytest` remains **DB-free**.
- Integration DB tests mirror ingestion: opt in with `LEGAL_RAG_RUN_INTEGRATION_DB=1` and `@pytest.mark.integration`.

## Next phase candidates

- Dense/sparse indexing and pgvector columns
- Manifest/version tables for reproducible index builds
- Optional HTTP ingest/chunk endpoints that call the same services as CLIs
