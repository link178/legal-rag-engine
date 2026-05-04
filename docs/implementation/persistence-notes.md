# Persistence notes — Phase 2A

## Why domain and persistence are separate

- **Domain models** (`app/domain/models/`) describe logical entities (`Document`, `Chunk`, `ProcessingRun`) without FastAPI, SQLAlchemy, or Alembic. They stay easy to unit test and portable.
- **ORM models** (`app/storage/postgres/models.py`) map rows in PostgreSQL. Repositories translate between domain objects and ORM rows.
- **Alembic** (`alembic.ini`, `migrations/`) owns schema evolution so environments do not drift.

## Tables in Phase 2A

| Table | Purpose |
|-------|---------|
| `processing_runs` | Minimal trace rows for operator-triggered ingest/index steps (status, timestamps, counters, optional error text). Not a job queue or SaaS workflow. |
| `documents` | Source identity, raw/normalized text, checksum, JSON metadata, optional **`created_by_run_id`** (FK → `processing_runs`, lineage for operator ingest). Unique on `(source_path, checksum)` for idempotent ingest. |
| `chunks` | Chunk text and lineage to `documents.id`; optional **`created_by_run_id`** (FK → `processing_runs`, lineage for chunking runs). Unique on `(document_id, chunk_index, chunking_strategy)`. |

Chunk body is stored in column **`text`**; the SQLAlchemy attribute is named **`chunk_text`** so it does not clash with SQLAlchemy’s ``text()`` helper.

JSON columns use **`metadata_json`** in the database (SQLAlchemy reserves `metadata` on declarative classes).

## Phase 2B (ingestion)

- Alembic revision adds `documents.created_by_run_id` → `processing_runs.id` (`ON DELETE SET NULL`). See [`ingestion-notes.md`](ingestion-notes.md).

## Phase 3 (chunk lineage)

- Alembic revision adds **`chunks.created_by_run_id`** → `processing_runs.id` (`ON DELETE SET NULL`). See [`chunking-notes.md`](chunking-notes.md).

## Tables in Phase 4A–4B (indexing)

| Table | Purpose |
|-------|---------|
| `index_manifests` | Snapshot of one indexing build: hashing, provider identity, dims, optional **`embedding_model`**, **`embeddings_persisted`**, counts, FK to **`processing_runs`** (`SET NULL` on delete). |
| `index_manifest_chunks` | Per-chunk row (`dense_indexed`, `sparse_indexed`, optional **`sparse_terms_json`**). No embedding blob here. |
| `chunk_embeddings` | **Phase 4B**: dense **`vector`** per chunk per manifest; FKs to **`chunks`** and **`index_manifests`** (`ON DELETE CASCADE`). Unique `(chunk_id, index_manifest_id)`. |

See [`indexing-notes.md`](indexing-notes.md).

## Future work (later phases)

- **Phase 5+**: pgvector **ANN** (HNSW/IVFFlat) once dimensions/index strategy are fixed; dense retrieval.
- **Sparse / FTS**: `tsvector` columns and GIN indexes, or separate sparse tables as designed.
- **API / ingestion**: wire repositories from ingestion and indexing services; keep `/health` liveness-only unless you add an explicit readiness endpoint.

## Commands

Default tests **do not** require Postgres:

```bash
pytest
```

When Docker Postgres is up and `DATABASE_URL` is correct:

```bash
docker compose up -d postgres
alembic upgrade head
alembic downgrade -1
```

Integration tests are marked `@pytest.mark.integration` and are **not** required for offline CI unless you opt in (see [`tests/integration/test_db_optional_placeholder.py`](../../tests/integration/test_db_optional_placeholder.py)).

## Docker init vs migrations

- `docker/postgres/init/01-enable-pgvector.sql` runs on **first** container DB creation and enables `vector`.
- The initial Alembic revision also runs `CREATE EXTENSION IF NOT EXISTS vector` so migrated databases stay consistent without relying only on init scripts.
