# Evaluation snapshot

UTC generation timestamp: **2026-07-29T14:00:00Z** (local validation after CI database isolation + explicit manifest gates)

Git commit at capture time: **`96d4c8e009f8503648fd0f1e907f23b988d6c9eb`** (`96d4c8e` on branch `develop`), with **uncommitted** CI isolation, explicit-manifest, and ranking-tie-break fixes in the working tree.

This file records checks executed in one local session. Metrics are copied from tool output only. No estimated figures.

## Historical baselines

### Pre-reliability-gate fix (same commit `96d4c8e`)

| Metric | Value | Notes |
|--------|--------|--------|
| Answer eval pass_rate | **0.6666666666666666** (2/3) | `q3_unknown_insufficient` returned `grounded` |
| Smoke pipeline | exit **0** | Gated only on `hit_rate > 0` and `pass_rate > 0` |
| Answer eval CLI | exit **0** | Did not gate on failed cases |
| Retrieval dataset | 3 questions | |
| Answer dataset | 3 questions | |

### Pre-isolation session (shared DB; auto-resolved / UUID-tied rankings)

Before database-level CI isolation and content-stable ranking tie-breaks, a contaminated or UUID-tied run could report **MRR = 0.7314814814814814**. That figure is **not** the current reproducible baseline.

## Current validated result (isolated reliability DB)

| Metric | Value |
|--------|--------|
| Default pytest (`python -m pytest -q`) | **413 passed**, **14 skipped** |
| Full DB-backed pytest (`LEGAL_RAG_RUN_INTEGRATION_DB=1` on integration-only DB) | **426 passed**, **1 skipped** (Windows symlink test) |
| Smoke pipeline (reliability-only DB) | exit **0**; retrieval `execution_status=PASSED`; answer `execution_status=PASSED` |
| Retrieval eval (explicit smoke manifest) | **9** questions; `hit_rate=1.0`; `MRR=0.75`; `execution_status=PASSED` |
| Answer eval (explicit smoke manifest) | **10** questions; `pass_rate=1.0` (10/10); `execution_status=PASSED`; `q3_unknown_insufficient` → `insufficient_context` |

### Reproducibility proof

| Run | Database | MRR |
|-----|----------|-----|
| A1 | fresh reliability DB | **0.75** |
| A2 | same DB, same explicit manifest, no writes | **0.75** |
| B1 | newly recreated reliability DB + smoke | **0.75** |

All three values match exactly.

### Contamination proof

After inserting a newer unrelated index manifest into the reliability DB, gated retrieval with `--index-manifest-id` set to the smoke-created UUID kept **hit_rate = 1.0**, **MRR = 0.75**, and the summary `manifest.id` equal to the smoke UUID (never the contaminant).

## Environment

| Item | Value |
|------|--------|
| OS | Windows 10 (win32 10.0.19045) |
| Python | 3.13.1 |
| pytest | 9.0.3 |
| ruff | 0.15.12 |
| mypy | 1.20.2 |
| Postgres | `pgvector/pgvector:pg16` on localhost:5432 (separate logical DBs for integration vs reliability) |

Install used: `pip install -e ".[dev,demo]"`.

## Checks

| Check | Status | Exit code | Notes |
|-------|--------|-----------|--------|
| `python -m pytest -q` | **PASSED** | 0 | Default local suite: 413 passed, 14 skipped (no DB opt-in) |
| Integration-only DB + `LEGAL_RAG_RUN_INTEGRATION_DB=1` + `python -m pytest -q` | **PASSED** | 0 | 426 passed, 1 skipped |
| `python -m ruff check .` | **PASSED** | 0 | Repository-wide Ruff |
| `python -m mypy` | **PASSED** | 0 | Configured production package scope (`app`) |
| `python -m mypy app` | **PASSED** | 0 | Equivalent explicit path |
| `python scripts/smoke_pipeline.py --manifest-out …` | **PASSED** | 0 | Writes smoke manifest UUID; evals use that id |
| Retrieval eval CLI (default gate, explicit manifest) | **PASSED** | 0 | See metrics below |
| Answer eval CLI (default gate, explicit manifest) | **PASSED** | 0 | See metrics below |
| Controlled failure-path proof | **PASSED** | — | Temporary wrong golden → `EXECUTED_WITH_FAILED_CASE`, CLI exit 1; `--no-gate` exit 0 |

Prerequisites for smoke/eval: `docker compose up -d postgres`, `alembic upgrade head` against the **reliability** database.

Mypy validates the configured production package scope (`app`). Do not treat `mypy .` as the official typing gate.

## CI database isolation

GitHub Actions runs three mandatory jobs:

1. **`lint-typecheck-test`** — Ruff, Mypy (`app`), default pytest (no Postgres).
2. **`integration-tests`** — dedicated Postgres DB; `LEGAL_RAG_RUN_INTEGRATION_DB=1`; full pytest.
3. **`reliability-gates`** — separate empty Postgres DB; migrate → smoke (`--manifest-out`) → gated retrieval/answer with `--index-manifest-id` from that file.

Integration-test artifacts cannot affect evaluation metrics because they never share a database with reliability gates. Evaluation metrics are produced from the explicit smoke-generated manifest only.

Remote GitHub Actions has **not** been re-run in this session.

## Smoke pipeline

Command:

```bash
python scripts/smoke_pipeline.py --manifest-out smoke_manifest_id.txt
```

Summary output (representative run; `manifest_id` is run-specific and not a durable metric):

```text
documents_chunked=7
evaluation_manifest_id=<smoke-created-uuid>
retrieval execution_status=PASSED hit_rate=1.0 total=9
answer execution_status=PASSED pass_rate=1.0 total=10
```

## Retrieval evaluation

Command (gated path pins the smoke manifest):

```bash
python -m app.evaluation.cli retrieval data/eval/retrieval_golden.jsonl \
  --mode hybrid --chunking-strategy fixed_size --top-k 5 \
  --index-manifest-id "$(tr -d ' \n' < smoke_manifest_id.txt)" --json
```

| Field | Value |
|-------|--------|
| Dataset | `data/eval/retrieval_golden.jsonl` (**9** questions) |
| Mode | `hybrid` |
| Chunking | `fixed_size` |
| top_k | 5 |
| Embedding | `deterministic_hash`, dimensions 16 |
| total / answered / errored | 9 / 9 / 0 |
| hit_rate | **1.0** |
| MRR | **0.75** |
| execution_status | **PASSED** |

Portfolio prose may round as **MRR: 0.75** (identical to the raw snapshot).

## Answer evaluation

Command:

```bash
python -m app.evaluation.cli answer data/eval/answer_golden.jsonl \
  --mode sparse_only --chunking-strategy fixed_size --provider mock \
  --index-manifest-id "$(tr -d ' \n' < smoke_manifest_id.txt)" --json
```

| Field | Value |
|-------|--------|
| Dataset | `data/eval/answer_golden.jsonl` (**10** questions) |
| Mode | `sparse_only` |
| Provider | `mock` |
| Chunking | `fixed_size` |
| top_k | 5 |
| total / answered / errored | 10 / 10 / 0 |
| pass_rate | **1.0** |
| mode_accuracy | 1.0 |
| expected_terms_accuracy | 1.0 |
| citation_validity_rate_avg | 0.6 |
| insufficient_context_accuracy | 1.0 |
| execution_status | **PASSED** |

`q3_unknown_insufficient`: expected `insufficient_context`, observed `insufficient_context` (`passed: true`).

## Failure-path validation (mandatory, not committed)

Temporary golden with mismatched expectations:

- `execution_status`: **EXECUTED_WITH_FAILED_CASE**
- CLI exit code (default gate): **1**
- CLI exit code with `--no-gate`: **0**

Proven for both retrieval and answer evaluation CLIs.

## Known limitations

- Metrics validate pipeline wiring on the committed sample corpus with `deterministic_hash` embeddings—not semantic retrieval quality at scale.
- Evidence sufficiency is lexical overlap, not entailment.
- Citation previews truncate at 240 characters (`GroundedCitation.text_preview`).
- Demo screenshot not yet committed; see README “Demo screenshot (TODO)”.
- Smoke `manifest_id` values are ephemeral per run and must not be treated as stable project metrics.
- Standalone evaluation CLI without `--index-manifest-id` still auto-resolves the latest suitable manifest (legacy/local convenience); the gated CI/reliability path always pins the smoke UUID.

## Reproduction notes

1. Use a **reliability-only** empty database (or clean reset) for smoke + gated eval; do not reuse an integration-test database.
2. Ingest via smoke (or ingest `data/sample_corpus/basic` + legalize sample) before eval goldens that reference those paths.
3. Always pass `--index-manifest-id` from smoke `--manifest-out` on the gated path.
4. Ranking tie-breaks use content keys (`source_path`, `chunk_index`) so equal scores do not depend on random chunk UUIDs across DB recreations.
