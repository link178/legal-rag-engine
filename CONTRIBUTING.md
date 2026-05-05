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

### Integration tests (opt-in)

Tests under `tests/integration/` are marked `@pytest.mark.integration`. They require a migrated PostgreSQL database and the environment variable:

```bash
export LEGAL_RAG_RUN_INTEGRATION_DB=1   # Windows: set LEGAL_RAG_RUN_INTEGRATION_DB=1
python -m pytest -q -m integration
```

## Linting and typing

```bash
python -m ruff check app tests
python -m mypy app
```

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
