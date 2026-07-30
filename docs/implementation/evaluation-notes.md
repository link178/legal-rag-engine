# Evaluation notes — Phase 5.5 (retrieval) & Phase 11 (answer)

## Goal

Phase 5.5 adds a **small, operator-controlled** harness to score **retrieval only** against a **golden JSONL** dataset. It measures whether retrieved chunks are **relevant by simple, auditable rules** before any generation phase.

Phase 11 adds **answer-level evaluation**: the same Postgres read path, but each golden row drives **`GroundedAnswerer`** → `GroundedAnswer`, then deterministic checks on **answer mode**, **expected terms**, **citation validity**, **insufficient-context flag**, and **whether cited sources hit expected paths**. Provider is **`mock` only** in this baseline (matches Phase 6/8).

Phase 11 does **not** add LLM-as-judge, semantic entailment, claim-by-claim verification, human eval, benchmarking against commercial APIs, dashboards, Streamlit, or HTTP surface changes.

This doc’s Phase 5.5 preamble does **not** cover `/v1/ask`; Phase 6 adds a mock-only generation CLI layered on retrieval; Phase 11 measures that path in bulk for regression-style checks.

## What it measures

- **Hit@k** (via `hit` + `hit_rank` in each item): at least one golden expectation is matched in a retrieved chunk within the first **k** results (k = CLI `--top-k` / `RetrievalConfig.top_k`).
- **MRR** (mean reciprocal rank): mean of `1/rank` for answered questions (errors contribute 0).

Matching is **first-match wins** by result order: the first retrieved chunk (within top_k) that satisfies **any** of the configured criteria counts as the hit; `matched_by` lists **all** criteria that matched **that** chunk.

Criteria (evaluated on each `RetrievedChunk`):

1. `chunk_id` ∈ `expected_chunk_ids`
2. `document_id` ∈ `expected_document_ids`
3. `source_path`: any `expected_source_paths` entry is a **case-insensitive substring** of `source_path` (or empty string if null)
4. **Terms**: **all** non-empty `expected_terms` must appear as **case-insensitive substrings** in whitespace-normalized chunk **text** (multi-word terms allowed; this is **not** the sparse retrieval tokenizer)

## What it does **not** measure

- Generation, grounding, or citation verification
- LLM-as-judge or embedding-similarity grading
- Reranking (placeholder remains in `app/retrieval/rerank.py`)
- Claim-level or legal-specific rubrics

## Read path and traceability

- Evaluation uses the same **Postgres read path** as Phase 5 (`RetrievalOrchestrator`, `resolve_manifest_record`, dense/sparse/RRF).
- **No new `processing_runs` rows** are created for evaluation (run_type `evaluation` is unused; invariant is 0 rows). Ingest/chunk/index remain the write-side traces.
- Reports include `schema_version: "evaluation.retrieval.v1"` plus `config` and `manifest` snapshots for reproducibility. **Phase 14:** `config["metadata_filter"]` is either `null` or the applied filter dict; the same filter is preserved when `RetrievalConfig` is rebuilt inside `run_questions` (regression-guarded). Per-question filters in golden JSONL are **deferred**.

## Golden JSONL format

One JSON object per non-empty line. Required: `id` (non-empty string), `question` (non-empty string), and **at least one** of:

- `expected_terms` (list of strings)
- `expected_document_ids` (list of UUID strings)
- `expected_chunk_ids` (list of UUID strings)
- `expected_source_paths` (list of strings)

Optional: `notes` (string).

Duplicate `id` values across lines are rejected at load time.

**Note:** `expected_chunk_ids` / `expected_document_ids` are only practical for a **fixed** corpus/database; the ship sample `data/eval/retrieval_golden.jsonl` uses **terms + source paths** only.

## CLI

Requires migrated Postgres, ingest, chunk, and indexing (same prerequisites as `app.retrieval.cli`).

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

Options mirror Phase 5 retrieval: `--mode dense_only|sparse_only|hybrid`, `--dense-top-k`, `--sparse-top-k`, `--rrf-k`, `--index-manifest-id`, `--embedding-provider`, `--embedding-model`, `--embedding-dimensions`, `--chunking-strategy`, plus **Phase 14** `--filter-*` metadata flags (see [`retrieval-notes.md`](retrieval-notes.md)).

Hybrid evaluation **requires** a manifest with `include_sparse=true` (re-index without `--no-sparse` if needed).

## Limitations (deterministic_hash)

Default `deterministic_hash` embeddings are **not** semantic retrieval. Dense scores are still computed, but **lexical/sparse and hybrid RRF** dominate interpretability for small corpora. Use this phase to validate **wiring and comparability**, not production relevance of dense-only mode.

## Tests

- Default `pytest`: unit tests only (no DB).
- Integration (retrieval eval): `LEGAL_RAG_RUN_INTEGRATION_DB=1` and `pytest -m integration` (`tests/integration/test_retrieval_evaluation_persist.py`, `tests/integration/test_retrieval_filter_persist.py`, `tests/integration/test_evaluation_manifest_isolation.py`).
- Integration (answer eval): same flag and `tests/integration/test_answer_evaluation_persist.py` (uses `sparse_only` + sample corpus).
- CI reliability gates use a **separate** Postgres database from integration pytest, write the smoke manifest via `--manifest-out`, and pin gated eval with `--index-manifest-id` (no auto-resolution on the gated path).

---

## Phase 11 — Answer evaluation (E2E mock)

### What it measures

- **Mode accuracy**: `GroundedAnswer.mode` vs optional golden `expected_mode` (`grounded` | `partial` | `insufficient_context`). If `expected_mode` is omitted, this check is a no-op (always matches).
- **Expected terms**: all non-empty `expected_terms` as case-insensitive substrings in whitespace-normalized text, controlled by `expected_terms_in`:
  - `answer` — answer body only
  - `citations` — `GroundedCitation.text_preview` only
  - `both` (default) — **either** answer **or** any citation preview (needed for `MockGenerationProvider`, whose answer text is fixed and domain-agnostic).
- **Citation validity**: from `CitationVerificationResult` on the answer (`citation_validity_rate`, `has_valid_citations`, `has_invalid_citations`).
- **Insufficient context**: `GroundedAnswer.insufficient_context` vs golden `should_be_insufficient_context`.
- **Cited source hit**: for non–`insufficient_context` answers, each non-empty `expected_source_paths` entry must appear as a **case-insensitive substring** of at least one **cited** `source_path` (`GroundedAnswer.citations`). For `insufficient_context`, this metric is **N/A** (`null` in JSON) and is excluded from `retrieved_expected_source_rate` averages.
- **Pass / fail**: per-item `passed` is true only when there is no runtime error and all applicable gates pass (see `compute_pass` in `app/evaluation/metrics/answer.py`).

### What it does **not** measure

- LLM-as-judge, embedding similarity, semantic faithfulness, legal rubrics
- Stability of numeric **citation ids** (1…N are positional in `ContextBuilder`; do **not** put `expected_citation_ids` in goldens)
- Full retrieval hit lists (use Phase 5.5 for Hit@k / MRR over raw hits)

### Golden JSONL (answer)

One JSON object per non-empty line. Required: `id`, `question`. Optional:

| Field | Meaning |
| --- | --- |
| `expected_mode` | `grounded` / `partial` / `insufficient_context` or omit |
| `expected_terms` | list of strings |
| `expected_terms_in` | `answer` / `citations` / `both` (default `both`) |
| `expected_source_paths` | list of path substrings vs **cited** sources |
| `should_be_insufficient_context` | boolean (default `false`) |
| `recommended_mode` | advisory: `dense_only` / `sparse_only` / `hybrid`; stderr warning if CLI mode differs |
| `notes` | free text |

Rows with unknown `expected_mode` or invalid `expected_terms_in` / `recommended_mode` are rejected at load time.

**Insufficient-context goldens** (e.g. off-corpus questions) rely on the deterministic evidence-sufficiency gate in `GroundedAnswerer` / `MockGenerationProvider`: retrieved chunks that do not materially support the question (stopword-aware content-term overlap) yield `insufficient_context` even when citation ids would otherwise be valid. The ship example `q3_unknown_insufficient` must remain `insufficient_context` under `sparse_only` (and typically under hybrid as well when evidence is insufficient).

### Execution status and exit codes

Completed evaluation reports include `summary.execution_status`:

| Status | Meaning |
| --- | --- |
| `PASSED` | Evaluation executed; every required case passed |
| `EXECUTED_WITH_FAILED_CASE` | Evaluation executed; one or more cases failed |
| `FAILED` | Evaluation could not execute reliably (empty/malformed results, load/config/runtime errors at the CLI boundary) |

By default the CLI **gates** on status: `PASSED` → exit 0; otherwise exit non-zero. Use `--no-gate` for informational runs that still exit 0 when cases failed but the harness executed (infrastructure failures remain non-zero). Smoke (`scripts/smoke_pipeline.py`) requires `execution_status == PASSED` for both retrieval and answer evaluation.

### CLI (answer)

Same DB prerequisites as `python -m app.generation.cli`. Subcommand:

```bash
python -m app.evaluation.cli answer data/eval/answer_golden.jsonl \
  --mode sparse_only \
  --chunking-strategy fixed_size \
  --top-k 5 \
  --provider mock \
  --json

python -m app.evaluation.cli answer data/eval/answer_golden.jsonl \
  --output answer-report.json \
  --markdown-output answer-report.md
```

Also supports explicit retrieval subcommand (equivalent to legacy):

```bash
python -m app.evaluation.cli retrieval data/eval/retrieval_golden.jsonl --json
```

**Legacy (unchanged):** `python -m app.evaluation.cli data/eval/retrieval_golden.jsonl --json` still runs retrieval evaluation when the first argument is not `answer` or `retrieval`.

Answer evaluation JSON uses `schema_version: "evaluation.answer.v1"`, `config` (including `max_chunks`, `max_context_chars`, `max_chunk_chars`, `min_score`, `provider`, and **Phase 14** `metadata_filter`), `manifest`, `summary` aggregates, and `items`. Markdown reports omit timestamps for stable diffs; they include a **Failures** section.

### Read path and traceability (Phase 11)

- Uses `GroundedAnswerer.from_session` (retrieval + `ContextBuilder` + mock provider + citation verification). **No** new `processing_runs` rows with `run_type="evaluation"`.
