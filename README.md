# legal-rag-engine

[![CI](https://github.com/link178/legal-rag-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/link178/legal-rag-engine/actions/workflows/ci.yml)
[![License: MIT](LICENSE)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169e1)
![tests](https://img.shields.io/badge/tests-pytest-green)
![ruff](https://img.shields.io/badge/lint-ruff-9333ea)
![mypy](https://img.shields.io/badge/types-mypy-blue)

**legal-rag-engine** is a **local-first**, **engine-first** hybrid RAG implementation with Python, FastAPI, PostgreSQL/pgvector, and Streamlit. It demonstrates document ingestion, fixed and structure-aware chunking, dense/sparse/hybrid retrieval with RRF fusion, grounded mock answers with mechanical citation verification, metadata-aware filters, and reproducible retrieval and answer evaluation.

## What this is

- Traceable ingestion (Markdown, TXT, HTML, PDF—text extraction only for PDF).
- Reproducible normalization, checksums, and processing runs.
- Comparable chunking strategies (`fixed_size`, `structure_aware`).
- Dense retrieval (pgvector), sparse retrieval (manifest-bound lexical maps), **hybrid** with RRF.
- Indexing manifests and optional `local_sentence_transformers` embeddings (`pip install -e ".[local-embeddings]"`).
- Grounded **mock** generation, citations, mechanical citation verification.
- Insufficient-context fallback.
- Golden JSONL evaluation for retrieval and answers (CLI).
- FastAPI `/v1/*` API and an HTTP-only Streamlit demo.

## What this is not

This is **not** legal advice, a compliance product, a SaaS, multi-tenant platform, AI Act assessment tool, or enterprise connector hub. The legal corpus adapter supports ingestion and retrieval experiments, not regulated legal workflows. See [docs/product/prd.md](docs/product/prd.md).

## Technical highlights

| Area | Notes |
|------|--------|
| Architecture | Engine and operator CLIs first; API is a thin wrapper. |
| Vector store | PostgreSQL **pgvector** default; Qdrant only as a future optional adapter. |
| Retrieval modes | `dense_only`, `sparse_only`, `hybrid` (not bare `dense` / `sparse`). |
| Indexing CLI | Dense and sparse are **on** by default; disable with `--no-dense` / `--no-sparse`. |
| Generation | **`mock`** only in this baseline; optional real providers belong behind extras later. |
| Demo | Streamlit calls **`/v1/*`** via `httpx`; no direct DB access from the UI. |

## Architecture

```text
Document Sources
    → Loaders / Parsers
    → Normalizer
    → Chunker
    → PostgreSQL documents/chunks
    → pgvector dense index
    → Sparse term maps (manifest-bound)
    → Dense / sparse / hybrid retriever + RRF
    → Context builder
    → Grounded generator (mock)
    → Citation verifier
    → API / Streamlit / Eval reports
```

## Quickstart

**Prerequisites:** Python **3.11+**, optionally [Docker](https://docs.docker.com/get-docker/) for Postgres + pgvector.

```bash
git clone <your-fork-or-mirror-url>
cd legal-rag-engine

python -m venv .venv
# Windows (PowerShell):  .\.venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate

pip install -e ".[dev,demo]"
```

Copy the env template (optional; defaults match `docker-compose.yml`):

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # macOS / Linux
```

Start Postgres and apply migrations:

```bash
docker compose up -d postgres
alembic upgrade head
```

Run tests (no Docker required):

```bash
python -m pytest -q
```

Quality gates used in development:

```bash
python -m ruff check app tests
python -m mypy app
```

## Full local smoke pipeline

**Prerequisites:** `docker compose up -d postgres`, `alembic upgrade head`, and `pip install -e ".[dev,demo]"`.

From the repo root:

```bash
python scripts/smoke_pipeline.py
```

This script is **operator-controlled**: it does not start Docker. It ingests `data/sample_corpus/basic` (so eval goldens resolve), imports `data/sample_corpus/legalize_sample`, chunks every persisted document with `fixed_size`, indexes, runs hybrid retrieval + mock generation on an EU-scoped question, then runs retrieval evaluation (`hybrid`) and answer evaluation (**`sparse_only`**, as required by the `answer_golden.jsonl` “Atlantis” case). It fails if any step errors or if golden `hit_rate` / `pass_rate` are zero.

Re-running against the same DB may hit idempotent skips (existing chunks/manifests); the script still collects the latest manifest id from indexing output.

## Run the API

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness (no DB check). |
| `GET` | `/` | Minimal metadata → `/docs`. |
| `POST` | `/v1/ingest` | Ingest a **server-local** path. |
| `GET` | `/v1/documents` | List documents. |
| `GET` | `/v1/documents/{document_id}` | Document detail. |
| `GET` | `/v1/documents/{document_id}/chunks` | Chunks for a document. |
| `POST` | `/v1/chunk` | Chunk a persisted document. |
| `POST` | `/v1/index` | Build index manifest + embeddings/sparse maps. |
| `GET` | `/v1/index-manifests` | List manifests. |
| `GET` | `/v1/index-manifests/{manifest_id}` | Manifest detail. |
| `POST` | `/v1/retrieve` | Dense / sparse / hybrid search. |
| `POST` | `/v1/answer` | Grounded answer (**`provider`: `mock` only**). |

**Not implemented:** `POST /v1/ask`, `POST /v1/evaluate`, multipart upload, auth.

Example (Linux / macOS) — ingest and pipeline (replace UUIDs from responses):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"path":"data/sample_corpus/basic/intro.md","persist":true}'
```

On **PowerShell**, `curl` may alias `Invoke-WebRequest`. Use **`Invoke-RestMethod`** or **`curl.exe`**:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
```

OpenAPI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Deep dive: [docs/implementation/api-notes.md](docs/implementation/api-notes.md).

## Run the Streamlit demo

Terminal 1 — API + DB:

```bash
docker compose up -d postgres
alembic upgrade head
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — UI:

```bash
streamlit run app/ui/streamlit_app.py
```

The demo exercises `/health`, ingestion, chunking, indexing, retrieval (optional **metadata filter**), and `/v1/answer` with **`mock`**. Details: [docs/implementation/demo-notes.md](docs/implementation/demo-notes.md).

## Evaluation

Goldens live under `data/eval/`. Use explicit subcommands:

```bash
python -m app.evaluation.cli retrieval data/eval/retrieval_golden.jsonl \
  --mode hybrid --chunking-strategy fixed_size --top-k 5 --json

python -m app.evaluation.cli answer data/eval/answer_golden.jsonl \
  --mode sparse_only --chunking-strategy fixed_size --provider mock --json
```

For stable runs when multiple manifests exist, pin **`--index-manifest-id`** (see indexing JSON output or `GET /v1/index-manifests`). More detail: [docs/implementation/evaluation-notes.md](docs/implementation/evaluation-notes.md).

## Legal corpus sample

Example tree: **`data/sample_corpus/legalize_sample/`** (`legalize-es`, `legalize-eu`). Import:

```bash
python -m app.ingestion.adapters.legal_corpus.cli data/sample_corpus/legalize_sample --persist --json
```

Notes: [docs/implementation/legal-corpus-notes.md](docs/implementation/legal-corpus-notes.md).

## Project structure (top level)

```text
app/           # Engine: ingestion, chunking, indexing, retrieval, generation, evaluation, api, ui
data/          # Sample corpora and eval goldens (committed fixtures)
docker/        # Postgres init (pgvector)
docs/          # PRD, ADR, implementation notes, learning guides
migrations/    # Alembic
scripts/       # Operator helpers (e.g. smoke_pipeline.py)
tests/         # Unit + opt-in integration DB tests
```

## Current status

**v0.1.0** — local-first RAG engine baseline (phases **1–14**): persistence, ingestion (incl. legal adapter), chunking, pgvector indexing with manifests, hybrid retrieval + RRF, metadata filters, mock grounded answers + citation checks, eval CLIs, FastAPI v1, Streamlit demo.

A detailed **phase-by-phase** digest (moved from an older README) lives in [docs/implementation/implementation-manual.md](docs/implementation/implementation-manual.md).

## Limitations

- **Mock generation** is the default; real LLM providers are future work behind optional extras.
- **PDF** support is text extraction only (**no OCR**).
- **Citation verification** is mechanical (IDs / snippets), not semantic entailment.
- **Sparse** retrieval in v1 is manifest-bound lexical over stored term maps; **Postgres FTS** as the sparse store is future scope.
- **Hybrid** requires indexing with sparse enabled (do not pass `--no-sparse` if you need hybrid).
- Benchmarks assume a migrated Postgres instance; default **`pytest`** does not.

## Roadmap (high level)

- Optional **real** generation / cloud embedding providers (extras-only; mock remains default).
- Richer sparse backends (e.g. Postgres FTS), rerankers, ANN tuning.
- Combined **`POST /v1/ask`** only if it stays a thin orchestration layer.

## Documentation

- Index: [docs/README.md](docs/README.md)
- PRD: [docs/product/prd.md](docs/product/prd.md)
- ADR-0001: [docs/decisions/0001-local-first-zero-cost-stack.md](docs/decisions/0001-local-first-zero-cost-stack.md)
- Portfolio-oriented copy: [docs/portfolio-summary.md](docs/portfolio-summary.md)

## Contributing / security / changelog

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [CHANGELOG.md](CHANGELOG.md)

## License

[MIT License](LICENSE).
