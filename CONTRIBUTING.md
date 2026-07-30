# Contributing

This project is an **engine-first**, **local-first** open-source RAG baseline aimed at portfolio and learning use. It is intentionally not a SaaS, compliance product, or multi-tenant platform.

## Local setup

From the repository root:

```bash
python -m venv .venv
# Windows (PowerShell): .\.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate

pip install -e ".[dev,demo]"
```

Optional extras:

- **Demo only** (Streamlit + httpx): `pip install -e ".[demo]"`
- **Local sentence-transformers embeddings**: `pip install -e ".[local-embeddings]"`

Copy environment template if needed:

```bash
cp .env.example .env   # or: copy .env.example .env on Windows
```

For retrieval / indexing against Postgres, start the database and run migrations (see [README.md](README.md)).

## Tests

Default suite (no Docker, no live DB):

```bash
python -m pytest -q
```

### Integration tests (opt-in, dedicated DB)

Tests under `tests/integration/` are marked `@pytest.mark.integration`. They require a migrated PostgreSQL database and:

```bash
export LEGAL_RAG_RUN_INTEGRATION_DB=1   # Windows: set LEGAL_RAG_RUN_INTEGRATION_DB=1
# Prefer a dedicated integration database (not the reliability DB):
export DATABASE_URL=postgresql+psycopg://legal_rag:legal_rag@localhost:5432/legal_rag_integration
alembic upgrade head
python -m pytest -q
```

### Reliability gates (separate DB)

Use a **separate empty database** so integration-test artifacts cannot affect metrics:

```bash
export DATABASE_URL=postgresql+psycopg://legal_rag:legal_rag@localhost:5432/legal_rag_reliability
# do not set LEGAL_RAG_RUN_INTEGRATION_DB
alembic upgrade head
python scripts/smoke_pipeline.py --manifest-out smoke_manifest_id.txt
# then gated eval with --index-manifest-id from that file (see README)
```

Evaluation metrics must come from the **explicit smoke-created manifest**. Unrelated manifests in the same DB must not be selected when `--index-manifest-id` is set.

## Linting and typing

```bash
python -m ruff check .  # repository-wide; mandatory
python -m mypy          # configured production package scope (`app`)
python -m mypy app      # equivalent explicit path
```

Mypy validates the configured production package scope (`app`). Do not treat `mypy .` as the official typing gate.

## CI layout

GitHub Actions (`.github/workflows/ci.yml`) runs three mandatory jobs in parallel:

1. **`lint-typecheck-test`** — Ruff, Mypy, default pytest (no Postgres).
2. **`integration-tests`** — isolated Postgres + full DB-backed pytest.
3. **`reliability-gates`** — isolated empty Postgres + smoke + gated retrieval/answer on the smoke manifest.

Jobs fail on command failure (`continue-on-error` is not used). Smoke and evaluation remain gated by default.

## Pull request guidelines

- Keep changes scoped to the bounded architecture: traceable ingestion, retrieval, grounded generation with mock provider by default, evaluation CLIs, thin API, Streamlit demo.
- Do not add mandatory paid external services or real LLM keys for the default baseline.
- Run `pytest`, `ruff`, and `mypy` before opening a PR.

## Scope boundaries

Out of scope for the public repo unless explicitly planned:

- Commercial compliance layer, AI Act assessment tooling, evidence packs.
- Auth, billing, SSO, organizations, background job runners for production SaaS.
- Breaking API changes without discussion and test updates.

For architecture and product intent, see [docs/product/prd.md](docs/product/prd.md) and [docs/decisions/0001-local-first-zero-cost-stack.md](docs/decisions/0001-local-first-zero-cost-stack.md).
