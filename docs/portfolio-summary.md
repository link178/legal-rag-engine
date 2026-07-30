# Portfolio summary — legal-rag-engine

## One-liner

Local-first hybrid RAG engine (Python, FastAPI, PostgreSQL/pgvector, Streamlit) with grounded mock answers, mechanical citation verification, and reproducible retrieval/answer evaluation.

## Problem

Many public RAG demos hide the hard parts: auditable ingestion, comparable chunking, reproducible indexes, explicit dense/sparse/hybrid retrieval, citation checks, deterministic evaluation, and controlled insufficient-context behavior. Recruiters and engineers need a readable reference that shows those practices without requiring paid APIs.

## Architecture

Engine-first pipeline: loaders (Markdown, TXT, HTML, PDF text extraction, `legalize-*` adapter) → normalization and checksums → PostgreSQL documents/chunks → chunking (`fixed_size` / `structure_aware`) → indexing manifests with pgvector dense embeddings and sparse term maps → dense / sparse / hybrid RRF retrieval with metadata filters → context builder → grounded mock generation → mechanical citation verification → FastAPI `/v1` and HTTP-only Streamlit demo. Evaluation CLIs reuse the same Postgres read path.

## Engineering decisions

- **Local-first, zero-cost baseline** (ADR-0001): Postgres/pgvector default; no API keys for smoke tests or unit tests.
- **Manifests** as the reproducibility unit for indexed state (provider, dimensions, sparse inclusion, chunking strategy).
- **Explicit retrieval modes** (`dense_only`, `sparse_only`, `hybrid`) rather than opaque “search.”
- **Mock generation + deterministic_hash embeddings** so CI and demos stay offline-reproducible; optional `local-embeddings` extra for sentence-transformers.
- **Thin delivery layer**: FastAPI and Streamlit wrap the engine; they do not own the domain logic.
- **Content-stable ranking tie-breaks** (`source_path`, `chunk_index`) so equal scores do not depend on random chunk UUIDs across database recreations.

## Reliability / evaluation approach

- Golden JSONL under `data/eval/` for retrieval (Hit@k, MRR) and answers (mode, terms, citations, insufficient-context).
- Operator smoke script: ingest → chunk → index → retrieve → generate → eval, with `--manifest-out` for CI handoff.
- CI: three mandatory jobs — repository-wide Ruff + Mypy on `app` + default pytest; **integration pytest on an isolated DB**; **reliability smoke + gated eval on a separate empty DB** using the explicit smoke-created `--index-manifest-id`. Integration-test artifacts cannot affect evaluation metrics.
- Streamlit demo uses pending pin queues (no post-widget session-state mutation); README embeds committed demo screenshots.

## Demonstrable result

**Current (committed on `develop`, CI green):** default pytest **422 passed** / **14 skipped**; CI integration pytest **436 passed** / **0 skipped** (Ubuntu); Windows local integration DB **435 passed** / **1 skipped** (symlink test); `python -m ruff check .` and `python -m mypy` (production package `app`) clean; smoke pipeline exit **0** with retrieval and answer `execution_status=PASSED`; retrieval golden (**9** questions, `hybrid`, top-k **5**) **hit_rate = 1.0**, **MRR = 0.75**; answer golden (**10** questions, `sparse_only`, `mock`) **pass_rate = 1.0** including `q3_unknown_insufficient` → `insufficient_context`. Latest CI run: [actions/runs/30532189588](https://github.com/link178/legal-rag-engine/actions/runs/30532189588). Full log: [evaluation-snapshot.md](evaluation-snapshot.md).

**Historical:** pre-fix answer pass_rate **0.667** (2/3) while smoke still exited 0; a pre-isolation MRR of **0.7314814814814814** is superseded by the reproducible **0.75** baseline after DB isolation and stable ranking ties.

## Limitations

- Default generation is **mock**; real LLM providers are out of scope for v0.1.0.
- PDFs: text extraction only; **no OCR**.
- Citation verification is **mechanical**, not entailment.
- Sparse retrieval is **manifest-bound** lexical in v1, not Postgres FTS.
- Dense baseline uses **`deterministic_hash`**, not semantic embeddings unless opted in.
- Not a compliance SaaS: no auth, multi-tenant layer, or AI Act product logic in-repo.
