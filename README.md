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

**Phase 2B — basic document ingestion.** `.txt` and Markdown (`.md`, `.markdown`) load into the domain `Document` via `app/ingestion/` (normalization + checksums). Operators can run `python -m app.ingestion.cli <file>`; optional `--persist` writes `documents` and `processing_runs` when Postgres is migrated. **Phase 8** adds `POST /v1/ingest` as a thin HTTP wrapper over the same pipeline (see [API notes](docs/implementation/api-notes.md)).

**Phase 3 — chunking.** Fixed-size and Markdown structure-aware strategies produce traceable domain `Chunk` rows. Optional Postgres persistence stores `chunks` with `created_by_run_id` and records a `processing_runs` row with `run_type="chunking"`. Operators run `python -m app.chunking.cli <document_uuid> --strategy fixed_size|structure_aware` or **`POST /v1/chunk`** (Phase 9). Still **no** Streamlit demo in the baseline repo path.

**Phase 4A — indexing foundations.** Deterministic pseudo-embeddings (no model downloads), sparse term-frequency maps, and manifest tables `index_manifests` / `index_manifest_chunks` trace indexing runs.

**Phase 4B — dense storage.** Vectors persist in **`chunk_embeddings`** (pgvector). Default provider **`deterministic_hash`**; optional **`local_sentence_transformers`** via `pip install -e ".[local-embeddings]"`. Operators run `python -m app.indexing.cli` after migrate + ingest + chunk (see [`docs/implementation/indexing-notes.md`](docs/implementation/indexing-notes.md)) or **`POST /v1/index`** (Phase 9).

**Phase 5 — retrieval.** Dense (pgvector L2), baseline lexical sparse over persisted term maps, hybrid with **RRF**, and operator CLI `python -m app.retrieval.cli`. **Phase 8** adds `POST /v1/retrieve` (read-only). See [`docs/implementation/retrieval-notes.md`](docs/implementation/retrieval-notes.md). ANN indexes, Postgres FTS, reranking, and Streamlit remain optional/future.

**Phase 5.5 — retrieval evaluation baseline.** Golden JSONL (e.g. `data/eval/retrieval_golden.jsonl`), Hit@k / MRR grading, JSON and optional Markdown reports, operator CLI `python -m app.evaluation.cli` (read-only DB; no new `processing_runs`). See [`docs/implementation/evaluation-notes.md`](docs/implementation/evaluation-notes.md).

**Phase 6 — grounded generation (mock).** `ContextBuilder`, grounded prompt assembly, `GenerationProvider` protocol, `MockGenerationProvider`, `GroundedAnswerer.from_session`, and operator CLI `python -m app.generation.cli` (requires ingest + chunk + index; read-only DB). **Phase 8** adds `POST /v1/answer` (mock only). See [`docs/implementation/generation-notes.md`](docs/implementation/generation-notes.md).

**Phase 7 — mechanical citation verification.** `verify_citations`, `CitationVerificationResult`, verification in `GroundedAnswerer`, and `citation_verification` in CLI JSON / HTTP answer response (same ingest/index prerequisites). Semantic claim verification and Streamlit remain future work.

**Phase 8 — API v1 (thin).** FastAPI routes under `/v1`: ingest, documents list/detail, retrieve, answer. Embedding family defaults from `Settings` (like CLIs). **Phase 9** adds chunk, index, manifests, and optional `GET …/documents/{id}/chunks`. **No** `/v1/ask`, **no** multipart upload, **no** auth. See [`docs/implementation/api-notes.md`](docs/implementation/api-notes.md).

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
- **Phases 8–9 — v1:** `POST /v1/ingest`, `GET /v1/documents`, `GET /v1/documents/{document_id}`, `GET /v1/documents/{document_id}/chunks`, `POST /v1/chunk`, `POST /v1/index`, `GET /v1/index-manifests`, `GET /v1/index-manifests/{manifest_id}`, `POST /v1/retrieve`, `POST /v1/answer` (mock generation only). See [`docs/implementation/api-notes.md`](docs/implementation/api-notes.md).

Recommended **HTTP pipeline** (same engine as CLI):

```bash
# 1. Ingest
curl -s -X POST http://127.0.0.1:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"data/sample_corpus/basic/intro.md\",\"persist\":true}"

# 2. Chunk (substitute document_id from step 1)
curl -s -X POST http://127.0.0.1:8000/v1/chunk \
  -H "Content-Type: application/json" \
  -d "{\"document_id\":\"<document_id>\",\"strategy\":\"fixed_size\",\"chunk_size\":1200,\"chunk_overlap\":200}"

# 3. Index (optional embedding_* fields fall back to server Settings when omitted)
curl -s -X POST http://127.0.0.1:8000/v1/index \
  -H "Content-Type: application/json" \
  -d "{\"chunking_strategy\":\"fixed_size\",\"include_dense\":true,\"include_sparse\":true}"

# 4. Inspect manifests (optional)
curl -s "http://127.0.0.1:8000/v1/index-manifests?limit=10&chunking_strategy=fixed_size"

# 5. Retrieve / Answer (pin index_manifest_id when multiple manifests exist)
curl -s -X POST http://127.0.0.1:8000/v1/retrieve \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What does the intro describe?\",\"mode\":\"hybrid\",\"chunking_strategy\":\"fixed_size\",\"top_k\":5,\"index_manifest_id\":\"<manifest_id>\"}"

curl -s -X POST http://127.0.0.1:8000/v1/answer \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What does the intro describe?\",\"mode\":\"hybrid\",\"chunking_strategy\":\"fixed_size\",\"top_k\":5,\"provider\":\"mock\",\"index_manifest_id\":\"<manifest_id>\"}"
```

The same steps work via **`app.chunking.cli`** and **`app.indexing.cli`** if you prefer terminals over HTTP—see chunked sections below.

**Not implemented:** `POST /v1/ask` (combined endpoint), `POST /v1/evaluate`, multipart upload.

### Phase 2B: ingest a file (CLI or API)

CLI (stdlib `argparse`):

```bash
python -m app.ingestion.cli data/sample_corpus/basic/intro.md
python -m app.ingestion.cli data/sample_corpus/basic/plain.txt --json
```

With Postgres up and `alembic upgrade head` applied (adds `documents.created_by_run_id` for run lineage):

```bash
python -m app.ingestion.cli data/sample_corpus/basic/intro.md --persist
```

HTTP (Phase 8): `POST /v1/ingest` with JSON `{"path":"<local-file>","persist":true}` — see [`docs/implementation/api-notes.md`](docs/implementation/api-notes.md).

Default **`pytest`** does **not** require Postgres. Targeted unit tests:

```bash
pytest tests/unit/test_ingestion_loaders.py
pytest tests/unit/test_ingestion_service.py
```

Details: [`docs/implementation/ingestion-notes.md`](docs/implementation/ingestion-notes.md).

### Phase 3: chunk a persisted document (HTTP or CLI)

Requires a migrated database (including `chunks.created_by_run_id`). Ingest with `--persist` first to obtain a `document_id`.

**HTTP (Phase 9):** `POST /v1/chunk` — see [`docs/implementation/api-notes.md`](docs/implementation/api-notes.md).

CLI (stdlib `argparse`):

```bash
python -m app.chunking.cli <document_uuid> --strategy fixed_size --json
python -m app.chunking.cli <document_uuid> --strategy structure_aware --json
```

Details: [`docs/implementation/chunking-notes.md`](docs/implementation/chunking-notes.md).

### Phase 4A–4B: index persisted chunks (HTTP or CLI)

After `alembic upgrade head`, ingest with `--persist`, then chunk at least one document.

**HTTP (Phase 9):** `POST /v1/index` — optional body fields mirror `app.indexing.cli`; omitted `embedding_*` values follow server `Settings`.

CLI (corpus-wide or per-strategy indexing; defaults from `EMBEDDING_PROVIDER`, `EMBEDDING_DIMENSIONS`, `EMBEDDING_MODEL` in `.env`):

```bash
python -m app.indexing.cli --chunking-strategy fixed_size --json
python -m app.indexing.cli --json
```

Optional real local embeddings (install extra first):

```bash
python -m app.indexing.cli --chunking-strategy fixed_size \
  --embedding-provider local_sentence_transformers \
  --embedding-model intfloat/multilingual-e5-small --embedding-dimensions 384 --json
```

See [`docs/implementation/indexing-notes.md`](docs/implementation/indexing-notes.md).

### Phase 5: retrieve chunks (CLI)

After indexing, with the same Postgres and manifest filters (defaults from `.env` when auto-selecting a manifest):

```bash
python -m app.retrieval.cli "your question" --mode hybrid --json
python -m app.retrieval.cli "keywords" --mode dense_only --top-k 5
```

Details: [`docs/implementation/retrieval-notes.md`](docs/implementation/retrieval-notes.md).

### Phase 5.5: evaluate retrieval (CLI)

After ingest, chunk, and indexing for the corpus under test (same Postgres + manifest filters as Phase 5):

```bash
python -m app.evaluation.cli data/eval/retrieval_golden.jsonl \
  --mode hybrid \
  --chunking-strategy fixed_size \
  --top-k 5 \
  --json

python -m app.evaluation.cli data/eval/retrieval_golden.jsonl \
  --output eval-report.json \
  --markdown-output eval-report.md
```

Details: [`docs/implementation/evaluation-notes.md`](docs/implementation/evaluation-notes.md).

### Phase 6: grounded answer (CLI)

After ingest, chunk, and indexing for the corpus under test (same Postgres + manifest filters as Phase 5):

```bash
python -m app.generation.cli "What does the basic intro file describe?" \
  --mode hybrid \
  --chunking-strategy fixed_size \
  --top-k 5 \
  --provider mock \
  --json
```

Details: [`docs/implementation/generation-notes.md`](docs/implementation/generation-notes.md).

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
- Indexing notes (Phases 4A–4B): [`docs/implementation/indexing-notes.md`](docs/implementation/indexing-notes.md)
- Retrieval notes (Phase 5): [`docs/implementation/retrieval-notes.md`](docs/implementation/retrieval-notes.md)
- Evaluation notes (Phase 5.5): [`docs/implementation/evaluation-notes.md`](docs/implementation/evaluation-notes.md)
- Generation notes (Phases 6–7): [`docs/implementation/generation-notes.md`](docs/implementation/generation-notes.md)
- API notes (Phase 8): [`docs/implementation/api-notes.md`](docs/implementation/api-notes.md)
- Learning guide: [`docs/learning/rag-learning-guide.md`](docs/learning/rag-learning-guide.md)
- System overview: [`docs/architecture/system-overview.md`](docs/architecture/system-overview.md)
- ADR-0001 (local-first zero-cost baseline): [`docs/decisions/0001-local-first-zero-cost-stack.md`](docs/decisions/0001-local-first-zero-cost-stack.md)

## Roadmap

1. **Bootstrap**: repo structure, config, logging, models and local Docker Compose.
2. **Persistence foundation (Phase 2A)**: domain models, SQLAlchemy + Alembic for `documents` / `chunks` / `processing_runs`, repositories (no ingestion API yet).
3. **Ingestion (Phase 2B)**: `.txt` / Markdown loaders, normalization, checksums, optional CLI persistence + run trace (`processing_runs`); **Phase 8** adds `POST /v1/ingest`.
4. **Chunking (Phase 3)**: fixed-size + structure-aware strategies, chunk metadata + checksums, optional persisted chunking + run trace (`chunks.created_by_run_id`); still no embeddings/retrieval/generation API.
5. **Indexing (Phases 4A–4B)**: deterministic + optional local embeddings; manifest tables; **`chunk_embeddings`** with pgvector; CLI (see indexing notes).
6. **Retrieval (Phase 5)**: dense pgvector search, baseline sparse over manifest term JSON, hybrid + RRF; operator CLI (see retrieval notes). Postgres FTS / ANN indexes optional later.
7. **Retrieval evaluation baseline (Phase 5.5)**: golden JSONL, Hit@k/MRR, JSON/Markdown reports; operator CLI (see evaluation notes). No generation or `/v1/evaluate`.
8. **Sparse retrieval storage (future)**: Postgres FTS / GIN as designed; Phase 5 sparse is manifest-bound lexical baseline only.
9. **Grounded generation (Phase 6)**: context builder + mock `GenerationProvider` + operator CLI (see generation notes). **Phase 8** adds `POST /v1/answer` (mock only). Combined `POST /v1/ask` and external LLMs remain future milestones.
10. **Mechanical citation verification (Phase 7)**: `verify_citations` + `GroundedAnswer.citation_verification` + CLI / HTTP answer JSON (see generation notes).
11. **API v1 thin layer (Phase 8)**: `POST /v1/retrieve` + `POST /v1/answer` + documents + ingest ([`api-notes.md`](docs/implementation/api-notes.md)); no `/v1/ask`, no chunk/index HTTP.
12. **End-to-end evaluation**: claim-level metrics, reproducible reports beyond retrieval-only mechanical checks.
13. **Demo and polish**: Streamlit demo, seed scripts, smoke tests and documentation cleanup.

## Explicit Exclusions

The public repo does not include proprietary AI Act taxonomy, commercial regulatory scoring, multi-tenant workflows, users/organizations, billing, SSO, premium evidence packs, enterprise dashboards, private connectors or advanced due-diligence logic.

Baseline stack boundaries are defined in ADR-0001 (linked under Documentation above).
