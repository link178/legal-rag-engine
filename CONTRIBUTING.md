# Contributing

This project is an **engine-first**, **local-first** open-source RAG baseline aimed at portfolio and learning use. It is intentionally not a SaaS, compliance product, or multi-tenant platform.

## Development workflow

1. Fork or clone the repository.
2. Create a focused branch from the current default branch (`develop`).
3. Keep the change small and scoped.
4. Add focused regression coverage for changed behaviour.
5. Run all applicable quality gates (see below).
6. Open a pull request with validation evidence.

Suggested branch naming:

```text
feat/<short-description>
fix/<short-description>
docs/<short-description>
test/<short-description>
chore/<short-description>
```

## Environment setup

**Prerequisites:** Python **3.11+**, optionally [Docker](https://docs.docker.com/get-docker/) for PostgreSQL/pgvector.

### Bash (macOS / Linux)

```bash
git clone https://github.com/link178/legal-rag-engine.git
cd legal-rag-engine

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,demo]"

cp .env.example .env

docker compose up -d postgres
python -m alembic upgrade head
```

### PowerShell (Windows)

```powershell
git clone https://github.com/link178/legal-rag-engine.git
cd legal-rag-engine

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev,demo]"

Copy-Item .env.example .env

docker compose up -d postgres
python -m alembic upgrade head
```

Optional extras:

- **Demo only** (Streamlit + httpx): `pip install -e ".[demo]"`
- **Local sentence-transformers embeddings**: `pip install -e ".[local-embeddings]"`

For retrieval, indexing, and DB-backed tests, Postgres must be running and migrations applied. See [README.md](README.md) for API, demo, and smoke-pipeline usage.

## Official quality gates

Run from the repository root after `pip install -e ".[dev,demo]"`:

```bash
python -m ruff check .
python -m mypy
python -m pytest -q
```

Mypy validates the configured production package scope, currently `app` (see `[tool.mypy]` in `pyproject.toml`). Do **not** treat `python -m mypy .` as the official passing gate.

GitHub Actions (`.github/workflows/ci.yml`) runs these three commands in the **`lint-typecheck-test`** job. That job does not require Postgres; DB integration tests are skipped unless opted in.

## DB-backed integration tests

Tests under `tests/integration/` are marked `@pytest.mark.integration`. They require a migrated, **isolated** PostgreSQL/pgvector database. Do **not** reuse the reliability-evaluation database.

Required environment variables:

```text
LEGAL_RAG_RUN_INTEGRATION_DB=1
DATABASE_URL=<isolated-integration-database-url>
```

Example URL (matches CI): `postgresql+psycopg://legal_rag:legal_rag@localhost:5432/legal_rag_integration`

### Bash

```bash
export LEGAL_RAG_RUN_INTEGRATION_DB=1
export DATABASE_URL=postgresql+psycopg://legal_rag:legal_rag@localhost:5432/legal_rag_integration

python -m alembic upgrade head
python -m pytest -q
```

### PowerShell

```powershell
$env:LEGAL_RAG_RUN_INTEGRATION_DB = "1"
$env:DATABASE_URL = "postgresql+psycopg://legal_rag:legal_rag@localhost:5432/legal_rag_integration"

python -m alembic upgrade head
python -m pytest -q
```

Each integration test starts from a clean database state (see `tests/conftest.py`).

## Reliability gates

Reliability validation mirrors the **`reliability-gates`** CI job. It must:

- use a **clean database isolated from integration tests** (do not set `LEGAL_RAG_RUN_INTEGRATION_DB`);
- run Alembic migrations;
- execute the smoke pipeline and capture the smoke-created index manifest;
- pass the explicit manifest ID to gated retrieval evaluation;
- pass the explicit manifest ID to answer evaluation;
- use **gated mode by default** (omit `--no-gate`; partial pass rates fail the gate);
- run with deterministic local fixtures and the **mock** provider;
- avoid external model APIs, paid services, and provider secrets.

Example reliability database URL (matches CI): `postgresql+psycopg://legal_rag:legal_rag@localhost:5432/legal_rag_reliability`

### Bash

```bash
export DATABASE_URL=postgresql+psycopg://legal_rag:legal_rag@localhost:5432/legal_rag_reliability

python -m alembic upgrade head
python scripts/smoke_pipeline.py --manifest-out smoke_manifest_id.txt

mid=$(tr -d ' \n' < smoke_manifest_id.txt)

python -m app.evaluation.cli retrieval data/eval/retrieval_golden.jsonl \
  --mode hybrid \
  --chunking-strategy fixed_size \
  --top-k 5 \
  --index-manifest-id "$mid" \
  --json

python -m app.evaluation.cli answer data/eval/answer_golden.jsonl \
  --mode sparse_only \
  --chunking-strategy fixed_size \
  --provider mock \
  --index-manifest-id "$mid" \
  --json
```

### PowerShell

```powershell
$env:DATABASE_URL = "postgresql+psycopg://legal_rag:legal_rag@localhost:5432/legal_rag_reliability"

python -m alembic upgrade head
python scripts/smoke_pipeline.py --manifest-out smoke_manifest_id.txt

$mid = (Get-Content smoke_manifest_id.txt -Raw).Trim()

python -m app.evaluation.cli retrieval data/eval/retrieval_golden.jsonl `
  --mode hybrid `
  --chunking-strategy fixed_size `
  --top-k 5 `
  --index-manifest-id $mid `
  --json

python -m app.evaluation.cli answer data/eval/answer_golden.jsonl `
  --mode sparse_only `
  --chunking-strategy fixed_size `
  --provider mock `
  --index-manifest-id $mid `
  --json
```

Do not rely on implicit manifest selection when running reliability gates. Unrelated manifests in the same database must not be selected when `--index-manifest-id` is set.

## Schema changes

Schema changes require:

- an Alembic migration under `migrations/`;
- applicable integration tests for persisted behaviour;
- validation from a clean migrated database;
- no manual schema drift outside migrations.

## Evaluation and documentation rules

Contributors must **not**:

- weaken evaluation thresholds to make CI green;
- remove failing regression cases without justification;
- label partial evaluation results as `PASSED`;
- manually publish metrics that were not regenerated;
- document metrics that differ from executed command output;
- rely on implicit manifest selection in reliability gates.

## Security and repository hygiene

Do **not** commit:

- `.env` files;
- API keys, tokens, credentials, or secrets;
- private, licensed, or confidential legal datasets;
- personal or confidential documents;
- local absolute filesystem paths;
- database dumps;
- temporary manifests (for example `smoke_manifest_id.txt`);
- controlled-failure golden files;
- pytest caches (`.pytest_cache/`);
- local virtual environments (`.venv/`);
- generated artifacts that are not intentionally documented.

## Pull request expectations

A pull request should include:

- a focused summary;
- behavioural impact;
- files changed;
- validation commands executed;
- exact test and evaluation results;
- known limitations;
- database or migration impact;
- confirmation that no secrets or local artifacts were added.

Keep changes scoped to the bounded architecture: traceable ingestion, retrieval, grounded generation with mock provider by default, evaluation CLIs, thin API, Streamlit demo. Do not add mandatory paid external services or real LLM keys for the default baseline.

## Pull request checklist

Before opening a pull request:

- [ ] The change is focused and documented.
- [ ] New or changed behaviour has regression coverage.
- [ ] `python -m ruff check .` passes.
- [ ] `python -m mypy` passes.
- [ ] `python -m pytest -q` passes.
- [ ] Applicable DB-backed integration tests pass.
- [ ] Schema changes include an Alembic migration.
- [ ] Reliability changes include smoke and evaluation evidence.
- [ ] Documentation metrics match executed command output.
- [ ] No secrets, private datasets, local paths, or temporary artifacts are committed.

## CI layout

GitHub Actions runs three mandatory jobs in parallel (no `continue-on-error`):

1. **`lint-typecheck-test`** — Ruff, Mypy, default pytest (no Postgres).
2. **`integration-tests`** — isolated Postgres + `LEGAL_RAG_RUN_INTEGRATION_DB=1` + full DB-backed pytest.
3. **`reliability-gates`** — isolated empty Postgres + smoke + gated retrieval/answer evaluation on the explicit smoke manifest.

## Scope boundaries

Out of scope for the public repo unless explicitly planned:

- Commercial compliance layer, AI Act assessment tooling, evidence packs.
- Auth, billing, SSO, organizations, background job runners for production SaaS.
- Breaking API changes without discussion and test updates.

For architecture and product intent, see [docs/product/prd.md](docs/product/prd.md) and [docs/decisions/0001-local-first-zero-cost-stack.md](docs/decisions/0001-local-first-zero-cost-stack.md).
