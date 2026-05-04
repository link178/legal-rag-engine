# Evaluation notes — Phase 5.5 (retrieval baseline)

## Goal

Phase 5.5 adds a **small, operator-controlled** harness to score **retrieval only** against a **golden JSONL** dataset. It measures whether retrieved chunks are **relevant by simple, auditable rules** before any generation phase.

This phase does **not** measure answer quality, full grounded generation with LLMs, citations as aligned with retrieval blocks, post-hoc citation verification, or `/v1/ask`/Streamlit demos (Phase 6 adds a mock-only generation CLI layered on retrieval; answer-quality evaluation remains separate).

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
- Reports include `schema_version: "evaluation.retrieval.v1"` plus `config` and `manifest` snapshots for reproducibility.

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

Options mirror Phase 5 retrieval: `--mode dense_only|sparse_only|hybrid`, `--dense-top-k`, `--sparse-top-k`, `--rrf-k`, `--index-manifest-id`, `--embedding-provider`, `--embedding-model`, `--embedding-dimensions`, `--chunking-strategy`.

Hybrid evaluation **requires** a manifest with `include_sparse=true` (re-index without `--no-sparse` if needed).

## Limitations (deterministic_hash)

Default `deterministic_hash` embeddings are **not** semantic retrieval. Dense scores are still computed, but **lexical/sparse and hybrid RRF** dominate interpretability for small corpora. Use this phase to validate **wiring and comparability**, not production relevance of dense-only mode.

## Tests

- Default `pytest`: unit tests only (no DB).
- Integration: `LEGAL_RAG_RUN_INTEGRATION_DB=1` and `pytest -m integration` (see `tests/integration/test_retrieval_evaluation_persist.py`).
