# legal-rag-engine

Local-first hybrid RAG and evaluation engine with FastAPI, PostgreSQL/pgvector, and Streamlit.

[![CI](https://github.com/link178/legal-rag-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/link178/legal-rag-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

**legal-rag-engine** is a local-first hybrid RAG pipeline: ingest documents, persist them with checksums and processing runs, chunk with comparable strategies, index dense (pgvector) and sparse (manifest-bound lexical maps) representations, retrieve with dense / sparse / hybrid RRF, and answer with grounded mock generation plus mechanical citation verification. Evaluation is first-class: golden JSONL datasets and operator CLIs measure retrieval Hit@k / MRR and answer pass rates against the same Postgres read path.

It is an engineering reference for how reliable RAG systems are built—traceability, comparable retrieval modes, grounded outputs, citation checks, deterministic evaluation, and controlled insufficient-context behavior—not legal advice, a compliance product, or a multi-tenant SaaS. See [docs/product/prd.md](docs/product/prd.md).

## Why this project

Public RAG demos often stop at “embed and chat.” Reliable retrieval and grounded answers depend on harder engineering: ingestion that can be audited, chunking that can be compared, indexes that are reproducible, retrieval modes that are explicit, generation that cites evidence, citation verification that is testable, evaluation that is deterministic, and a clear path when context is insufficient. This repository makes those pieces concrete and runnable locally without paid APIs.

## What it demonstrates

- Engine-first design: domain services and operator CLIs first; FastAPI and Streamlit are thin consumers.
- Traceable ingestion for Markdown, TXT, HTML, and PDF (text extraction only), plus a `legalize-*` corpus adapter.
- Comparable chunking (`fixed_size`, `structure_aware`) with config hashing.
- Dense retrieval (pgvector), sparse retrieval (manifest-bound term maps), and hybrid fusion via Reciprocal Rank Fusion (RRF).
- Indexing manifests as the unit of reproducibility for “what was indexed.”
- Grounded **mock** generation, mechanical citation verification, and insufficient-context fallback.
- Deterministic golden evaluation for retrieval and answers (CLI), plus CI gates (repository-wide Ruff, Mypy on `app`, isolated integration pytest DB, and a separate reliability DB for smoke + gated evaluation on the explicit smoke manifest).

## Architecture

```mermaid
flowchart LR
  subgraph sources [Sources]
    MD[Markdown / TXT]
    HTML[HTML]
    PDF[PDF text extract]
    LEGAL[legalize-* adapter]
  end

  subgraph ingest [Ingestion]
    LOAD[Loaders / parsers]
    NORM[Normalizer + checksums]
  end

  subgraph store [PostgreSQL]
    DOCS[(documents / chunks)]
    RUNS[(processing_runs)]
    DENSE[(pgvector embeddings)]
    SPARSE[(sparse term maps)]
    MAN[(index_manifests)]
  end

  subgraph retrieve [Retrieval]
    DRET[Dense retriever]
    SRET[Sparse retriever]
    RRF[Hybrid RRF fusion]
    FILT[Metadata filters]
  end

  subgraph generate [Generation]
    CTX[Context builder]
    GEN[Grounded generator mock]
    CIT[Citation verifier]
  end

  subgraph delivery [Consumers]
    API[FastAPI /v1]
    UI[Streamlit demo]
    EVAL[Eval CLIs / reports]
  end

  MD --> LOAD
  HTML --> LOAD
  PDF --> LOAD
  LEGAL --> LOAD
  LOAD --> NORM
  NORM --> DOCS
  NORM --> RUNS
  DOCS --> CHUNK[Chunker fixed_size / structure_aware]
  CHUNK --> DOCS
  DOCS --> IDX[Indexer]
  IDX --> DENSE
  IDX --> SPARSE
  IDX --> MAN
  MAN --> DRET
  MAN --> SRET
  DENSE --> DRET
  SPARSE --> SRET
  DRET --> RRF
  SRET --> RRF
  FILT --> DRET
  FILT --> SRET
  RRF --> CTX
  DRET --> CTX
  SRET --> CTX
  CTX --> GEN
  GEN --> CIT
  CIT --> API
  CIT --> UI
  CIT --> EVAL
  RRF --> EVAL
```

Text equivalent of the same pipeline:

```text
Document sources → loaders / parsers → normalizer
  → PostgreSQL documents / chunks / processing_runs
  → chunker (fixed_size | structure_aware)
  → indexer → pgvector dense + sparse term maps + index manifest
  → dense / sparse / hybrid (RRF) + metadata filters
  → context builder → grounded mock generator → citation verifier
  → FastAPI / Streamlit / evaluation reports
```

## Evaluation snapshot

Evidence from the latest local validation session (CI DB isolation + explicit smoke manifest; base commit `96d4c8e`). Full command log and historical baselines: [docs/evaluation-snapshot.md](docs/evaluation-snapshot.md).

| Check | Result |
|-------|--------|
| Default pytest (`python -m pytest -q`, no DB opt-in) | **413 passed**, **14 skipped** |
| Full DB-backed pytest (integration-only DB, `LEGAL_RAG_RUN_INTEGRATION_DB=1`) | **426 passed**, **1 skipped** (Windows symlink test) |
| Ruff (`python -m ruff check .`) | **passed** |
| Mypy (`python -m mypy` / `python -m mypy app`) | **passed** — validates the configured production package scope (`app`) |
| Smoke pipeline (`scripts/smoke_pipeline.py --manifest-out …`) | **passed** (exit 0); 7 documents chunked; retrieval and answer `execution_status=PASSED` |
| Retrieval eval | dataset **9** questions; mode `hybrid`; chunking `fixed_size`; top-k **5**; embedding `deterministic_hash` (16-d); **hit_rate = 1.0**; **MRR = 0.75**; **execution_status = PASSED** (explicit smoke manifest) |
| Answer eval | dataset **10** questions; mode `sparse_only`; provider `mock`; **pass_rate = 1.0** (10/10); **`q3_unknown_insufficient` → insufficient_context**; **execution_status = PASSED** (explicit smoke manifest) |

**CI isolation:** integration pytest and reliability smoke/eval use **separate databases**. Metrics are produced only from the smoke-created manifest (`--manifest-out` → `--index-manifest-id`). Integration-test artifacts cannot affect evaluation metrics.

**Pre-fix baseline** (same commit before reliability/isolation work): answer pass_rate **0.667** (2/3) with smoke exit 0 despite a failed golden case. A pre-isolation MRR of **0.7314814814814814** is historical only—not the current reproducible value.

These figures use a local Postgres instance after `alembic upgrade head`. They are wiring and regression evidence on the sample corpus—not a large-benchmark claim. Dense embeddings in this baseline are `deterministic_hash`, not semantic models.

## Main capabilities

| Area | Notes |
|------|--------|
| Architecture | Engine and operator CLIs first; API is a thin wrapper. |
| Vector store | PostgreSQL **pgvector** default; Qdrant only as a future optional adapter. |
| Retrieval modes | `dense_only`, `sparse_only`, `hybrid` (not bare `dense` / `sparse`). |
| Indexing CLI | Dense and sparse are **on** by default; disable with `--no-dense` / `--no-sparse`. |
| Generation | **`mock`** only in this baseline; optional real providers belong behind extras later. |
| Demo | Streamlit calls **`/v1/*`** via `httpx`; no direct DB access from the UI. |
| Filters | Exact-match metadata filters on document JSON (retrieval, generation, evaluation, Streamlit). |

## Quickstart

**Prerequisites:** Python **3.11+**, optionally [Docker](https://docs.docker.com/get-docker/) for Postgres + pgvector.

```bash
git clone https://github.com/link178/legal-rag-engine.git
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

Default local suite (no Docker required; DB integration tests are skipped without opt-in):

```bash
python -m pytest -q
```

Full DB-backed suite (requires Postgres + migrations; prefer a dedicated integration database so reliability metrics stay clean):

```bash
# Windows PowerShell:
$env:LEGAL_RAG_RUN_INTEGRATION_DB = "1"
$env:DATABASE_URL = "postgresql+psycopg://legal_rag:legal_rag@localhost:5432/legal_rag_integration"
python -m alembic upgrade head
python -m pytest -q
```

Quality gates used in development and CI:

```bash
python -m ruff check .   # repository-wide; mandatory
python -m mypy           # configured production package scope (`app`)
python -m mypy app       # equivalent explicit path
```

### Full local smoke pipeline

**Prerequisites:** `docker compose up -d postgres`, `alembic upgrade head` on a **reliability-only** (or clean) database, and `pip install -e ".[dev,demo]"`. Do not run smoke/gated eval against a database that just ran integration pytest.

From the repo root:

```bash
python scripts/smoke_pipeline.py --manifest-out smoke_manifest_id.txt
```

Then gated evaluation (same explicit manifest):

```bash
# Windows PowerShell example:
$mid = (Get-Content smoke_manifest_id.txt -Raw).Trim()
python -m app.evaluation.cli retrieval data/eval/retrieval_golden.jsonl --mode hybrid --chunking-strategy fixed_size --top-k 5 --index-manifest-id $mid --json
python -m app.evaluation.cli answer data/eval/answer_golden.jsonl --mode sparse_only --chunking-strategy fixed_size --provider mock --index-manifest-id $mid --json
```

This script is **operator-controlled**: it does not start Docker. It ingests `data/sample_corpus/basic` (so eval goldens resolve), imports `data/sample_corpus/legalize_sample`, chunks every persisted document with `fixed_size`, indexes, runs hybrid retrieval + mock generation on an EU-scoped question, then runs retrieval evaluation (`hybrid`) and answer evaluation (`sparse_only`, `mock`) against the **same** created manifest. It fails if any step errors or if evaluation `execution_status` is not `PASSED` (partial pass rates such as 0.667 are failures).

## API and demo usage

### Run the API

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

### Run the Streamlit demo

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

### Streamlit demo

The Streamlit client exercises the FastAPI pipeline end to end using an
explicitly pinned index manifest. This example performs hybrid retrieval,
returns a grounded deterministic answer, and mechanically verifies all used
citation identifiers.

![Streamlit demo showing a grounded hybrid answer](docs/assets/demo-answer-overview.png)

The verification output distinguishes used, available, valid, invalid,
duplicate, and unused citation identifiers. In this execution, all three used
citations were valid and no invalid citations were produced.

![Mechanical citation verification with a 1.0 validity rate](docs/assets/demo-citation-verification.png)

## Reproducible evaluation

Goldens live under `data/eval/` (9 retrieval questions, 10 answer questions). Use explicit subcommands after ingest → chunk → index on a migrated Postgres instance:

```bash
python -m app.evaluation.cli retrieval data/eval/retrieval_golden.jsonl \
  --mode hybrid --chunking-strategy fixed_size --top-k 5 --json

python -m app.evaluation.cli answer data/eval/answer_golden.jsonl \
  --mode sparse_only --chunking-strategy fixed_size --provider mock --json
```

For stable runs when multiple manifests exist, pin **`--index-manifest-id`** (see indexing JSON output or `GET /v1/index-manifests`). Or run the full operator path:

```bash
python scripts/smoke_pipeline.py
```

More detail: [docs/implementation/evaluation-notes.md](docs/implementation/evaluation-notes.md) and [docs/evaluation-snapshot.md](docs/evaluation-snapshot.md).

### Legal corpus sample

Example tree: **`data/sample_corpus/legalize_sample/`** (`legalize-es`, `legalize-eu`). Import:

```bash
python -m app.ingestion.adapters.legal_corpus.cli data/sample_corpus/legalize_sample --persist --json
```

Notes: [docs/implementation/legal-corpus-notes.md](docs/implementation/legal-corpus-notes.md).

## Project structure

```text
app/           # Engine: ingestion, chunking, indexing, retrieval, generation, evaluation, api, ui
data/          # Sample corpora and eval goldens (committed fixtures)
docker/        # Postgres init (pgvector)
docs/          # PRD, ADR, implementation notes, evaluation snapshot, learning guides
migrations/    # Alembic
scripts/       # Operator helpers (e.g. smoke_pipeline.py)
tests/         # Unit + opt-in integration DB tests
```

**v0.1.0** — local-first RAG engine baseline (phases **1–14**): persistence, ingestion (incl. legal adapter), chunking, pgvector indexing with manifests, hybrid retrieval + RRF, metadata filters, mock grounded answers + citation checks, eval CLIs, FastAPI v1, Streamlit demo.

A detailed phase-by-phase digest lives in [docs/implementation/implementation-manual.md](docs/implementation/implementation-manual.md).

## Deliberate limitations

- **Mock generation** is the default; real LLM providers are future work behind optional extras.
- **PDF** support is text extraction only (**no OCR**).
- **Citation verification** is mechanical (IDs / snippets), not semantic entailment.
- **Sparse** retrieval in v1 is manifest-bound lexical over stored term maps; **Postgres FTS** as the sparse store is future scope.
- **Hybrid** requires indexing with sparse enabled (do not pass `--no-sparse` if you need hybrid).
- Default dense embeddings are **`deterministic_hash`** (reproducible wiring), not a semantic embedding model unless you opt into `local-embeddings`.
- Benchmarks assume a migrated Postgres instance; default **`pytest`** does not run integration DB tests.
- Intentionally out of scope for this public baseline: auth, multi-tenant SaaS, `POST /v1/ask`, upload APIs, and compliance / AI Act product logic.

## Roadmap

- Optional **real** generation / cloud embedding providers (extras-only; mock remains default).
- Richer sparse backends (e.g. Postgres FTS), rerankers, ANN tuning.
- Combined **`POST /v1/ask`** only if it stays a thin orchestration layer.

## Documentation

- Index: [docs/README.md](docs/README.md)
- Evaluation snapshot: [docs/evaluation-snapshot.md](docs/evaluation-snapshot.md)
- PRD: [docs/product/prd.md](docs/product/prd.md)
- ADR-0001: [docs/decisions/0001-local-first-zero-cost-stack.md](docs/decisions/0001-local-first-zero-cost-stack.md)
- Portfolio-oriented copy: [docs/portfolio-summary.md](docs/portfolio-summary.md)

## Contributing / security / changelog

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [CHANGELOG.md](CHANGELOG.md)

## License

[MIT License](LICENSE).
