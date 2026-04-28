# Legal RAG Engine

**Legal RAG Engine** is a public, open-source, **engine-first** RAG project for building and evaluating grounded question-answering over document corpora.

It is designed to demonstrate serious RAG engineering on general document collections and legal corpora such as `legalize-*`, while keeping the v1 baseline **local-first** and runnable without mandatory paid services.

## What It Is

A reusable RAG engine focused on:

- traceable document ingestion;
- reproducible normalization;
- comparable chunking strategies;
- dense retrieval;
- sparse retrieval;
- hybrid retrieval with RRF;
- grounded generation with citations;
- citation verification;
- insufficient context fallback;
- reproducible evaluation;
- a FastAPI API and simple Streamlit demo.

## What It Is Not

This repository is not:

- a SaaS product;
- a multi-tenant platform;
- an AI Act compliance product;
- a commercial evidence-pack system;
- an enterprise dashboard;
- a billing, SSO or organization-management layer;
- a private connector hub;
- an advanced legal due-diligence product.

The legal corpus use case is important, but the engine remains reusable for general document corpora.

## Status

**Phase 1 — technical bootstrap.** The repo has a runnable FastAPI app with `/health`, centralized settings, logging, and local PostgreSQL + pgvector via Docker Compose.

**Phase 2A — persistence foundation.** Domain models (`Document`, `Chunk`, `ProcessingRun`), SQLAlchemy mappings, Alembic migrations for `documents`, `chunks`, and `processing_runs`, and session/repository helpers live under `app/domain/` and `app/storage/postgres/`.

**Phase 2B — basic document ingestion.** `.txt` and Markdown (`.md`, `.markdown`) load into the domain `Document` via `app/ingestion/` (normalization + checksums). Operators can run `python -m app.ingestion.cli <file>`; optional `--persist` writes `documents` and `processing_runs` when Postgres is migrated. There is still **no** `POST /v1/ingest`, embeddings, retrieval, or generation wired to the API.

**Phase 3 — chunking.** Fixed-size and Markdown structure-aware strategies produce traceable domain `Chunk` rows. Optional Postgres persistence stores `chunks` with `created_by_run_id` and records a `processing_runs` row with `run_type="chunking"`. Operators run `python -m app.chunking.cli <document_uuid> --strategy fixed_size|structure_aware`. Still **no** real embeddings, dense/sparse retrieval, or Streamlit demo.

**Phase 4A — indexing foundations.** Deterministic pseudo-embeddings (no model downloads), sparse term-frequency maps, and manifest tables `index_manifests` / `index_manifest_chunks` trace indexing runs. Operators run `python -m app.indexing.cli` after migrate + ingest + chunk. Retrieval, generation, `/v1/*` API, and pgvector storage on `chunks` remain out of scope until Phase 4B+ (see [`docs/implementation/indexing-notes.md`](docs/implementation/indexing-notes.md)).

Retrieval, generation, Streamlit demo, and evaluation wiring beyond chunk persistence remain **scaffolds or placeholders** until later phases.

## Phase 1: run locally

### Prerequisites

- Python **3.11+**
- [Docker](https://docs.docker.com/get-docker/) (optional, for Postgres + pgvector)

### Install

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / macOS

pip install -e ".[dev]"
```

Copy environment template (optional):

```bash
copy .env.example .env   # Windows
# cp .env.example .env     # Linux / macOS
```

### PostgreSQL + pgvector (Docker)

From the repository root:

```bash
docker compose up -d postgres
```

This starts Postgres on port **5432** with user/database `legal_rag` and enables the `vector` extension via `docker/postgres/init/`. The API **does not** require Postgres to be running for default tests or for `GET /health`.

### Database migrations (when Postgres is available)

Schema is managed by **Alembic** (`alembic.ini`, `migrations/`). With `DATABASE_URL` set (see `.env.example`) and Postgres running:

```bash
alembic upgrade head
# optional:
alembic downgrade -1
```

Migrations apply versioned DDL (including `CREATE EXTENSION IF NOT EXISTS vector` in the initial revision). They are **not** required for offline `pytest`.

### API

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health: `GET /health` → JSON with `status`, `service`, `environment`
- Root: `GET /` → minimal metadata and pointer to `/docs`

**Not implemented yet:** `POST /v1/ingest`, `GET /v1/documents`, `POST /v1/ask`, `POST /v1/evaluate`.

### Phase 2B: ingest a file (CLI)

No HTTP API for ingest in this phase; use the module CLI (stdlib `argparse`, no extra deps):

```bash
python -m app.ingestion.cli data/sample_corpus/basic/intro.md
python -m app.ingestion.cli data/sample_corpus/basic/plain.txt --json
```

With Postgres up and `alembic upgrade head` applied (adds `documents.created_by_run_id` for run lineage):

```bash
python -m app.ingestion.cli data/sample_corpus/basic/intro.md --persist
```

Default **`pytest`** does **not** require Postgres. Targeted unit tests:

```bash
pytest tests/unit/test_ingestion_loaders.py
pytest tests/unit/test_ingestion_service.py
```

Details: [`docs/implementation/ingestion-notes.md`](docs/implementation/ingestion-notes.md).

### Phase 3: chunk a persisted document (CLI)

Requires a migrated database (including `chunks.created_by_run_id`). Ingest with `--persist` first to obtain a `document_id`:

```bash
python -m app.chunking.cli <document_uuid> --strategy fixed_size --json
python -m app.chunking.cli <document_uuid> --strategy structure_aware --json
```

Details: [`docs/implementation/chunking-notes.md`](docs/implementation/chunking-notes.md).

### Phase 4A: index persisted chunks (CLI)

After `alembic upgrade head`, ingest with `--persist`, then chunk a document. Corpus-wide or per-strategy indexing:

```bash
python -m app.indexing.cli --chunking-strategy fixed_size --json
python -m app.indexing.cli --json
```

See [`docs/implementation/indexing-notes.md`](docs/implementation/indexing-notes.md).

### Tests and quality

```bash
pytest
ruff check app tests
mypy app
```

Tests are designed to run **without** Docker, Postgres, external API keys, embeddings, or LLMs. Integration DB tests under `tests/integration/` are marked `integration` and skip unless you opt in (see [`docs/implementation/persistence-notes.md`](docs/implementation/persistence-notes.md)).

## Target Stack

Default v1 stack:

- Python 3.11+ / 3.12;
- FastAPI;
- Pydantic v2;
- Typer for CLI;
- PostgreSQL;
- pgvector as the default vector backend;
- Postgres Full Text Search and/or local sparse retrieval;
- Streamlit for the initial demo;
- Docker Compose for local development.

Qdrant is **not** part of the baseline. It may be added later as an **optional adapter**, not as the recommended default vector store.

## Local-First Zero-Cost Baseline

The v1 baseline should run locally with a small example corpus. Basic tests, smoke tests and the demo must not require paid external services or mandatory API keys.

Recommended defaults:

- local embeddings when possible;
- mock generation for tests;
- optional local generation through tools such as Ollama;
- external embedding/LLM providers only behind provider interfaces.

## Pipeline

```text
Document Sources
    → Loaders / Parsers
    → Normalizer
    → Chunker
    → PostgreSQL documents/chunks
    → pgvector dense index
    → Postgres FTS / sparse local index
    → Dense/Sparse/Hybrid Retriever
    → RRF Fusion
    → Optional Reranker
    → Context Builder
    → Grounded Generator
    → Citation Verifier
    → API / Streamlit Demo / Eval Reports
```

## Documentation

- Documentation index: [`docs/README.md`](docs/README.md)
- Product PRD: [`docs/product/prd.md`](docs/product/prd.md)
- Implementation manual: [`docs/implementation/implementation-manual.md`](docs/implementation/implementation-manual.md)
- Persistence notes (Phase 2A): [`docs/implementation/persistence-notes.md`](docs/implementation/persistence-notes.md)
- Ingestion notes (Phase 2B): [`docs/implementation/ingestion-notes.md`](docs/implementation/ingestion-notes.md)
- Chunking notes (Phase 3): [`docs/implementation/chunking-notes.md`](docs/implementation/chunking-notes.md)
- Indexing notes (Phase 4A): [`docs/implementation/indexing-notes.md`](docs/implementation/indexing-notes.md)
- Learning guide: [`docs/learning/rag-learning-guide.md`](docs/learning/rag-learning-guide.md)
- System overview: [`docs/architecture/system-overview.md`](docs/architecture/system-overview.md)
- ADR-0001 (local-first zero-cost baseline): [`docs/decisions/0001-local-first-zero-cost-stack.md`](docs/decisions/0001-local-first-zero-cost-stack.md)

## Roadmap

1. **Bootstrap**: repo structure, config, logging, models and local Docker Compose.
2. **Persistence foundation (Phase 2A)**: domain models, SQLAlchemy + Alembic for `documents` / `chunks` / `processing_runs`, repositories (no ingestion API yet).
3. **Ingestion (Phase 2B)**: `.txt` / Markdown loaders, normalization, checksums, optional CLI persistence + run trace (`processing_runs`); still no `/v1/ingest`.
4. **Chunking (Phase 3)**: fixed-size + structure-aware strategies, chunk metadata + checksums, optional persisted chunking + run trace (`chunks.created_by_run_id`); still no embeddings/retrieval/generation API.
5. **Indexing foundations (Phase 4A)**: deterministic embedding interface + sparse term maps + manifest tables; CLI; no pgvector column on chunks yet (see indexing notes).
6. **Dense/sparse indexing (Phase 4B+)**: local embeddings, pgvector, Postgres FTS and/or sparse retrieval storage.
7. **Hybrid retrieval**: dense retrieval, sparse retrieval, RRF fusion and comparable modes.
8. **Grounded generation**: context builder, mock/local generation, citations and insufficient context fallback.
9. **Evaluation**: golden dataset, retrieval metrics, citation checks and reproducible reports.
10. **Demo and polish**: Streamlit demo, seed scripts, smoke tests and documentation cleanup.

## Explicit Exclusions

The public repo does not include proprietary AI Act taxonomy, commercial regulatory scoring, multi-tenant workflows, users/organizations, billing, SSO, premium evidence packs, enterprise dashboards, private connectors or advanced due-diligence logic.

Baseline stack boundaries are defined in ADR-0001 (linked under Documentation above).
