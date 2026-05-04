# Generation notes — Phase 6 (mock grounded QA)

## Goal

Phase 6 wires **operator-controlled grounded answering** after retrieval:

```text
question
  → RetrievalOrchestrator.retrieve (same contract as Phase 5)
  → ContextBuilder (ranked snippets → citation blocks [1],[2],…)
  → build_grounded_prompt(full prompt string)
  → GenerationProvider.generate(prompt) → raw answer text
  → extract citation ids [\d+] from raw answer vs context blocks
  → GroundedAnswer (mode + citations metadata)
```

There is **no** `POST /v1/ask`, **no** OpenAI/Ollama, **no** streaming, **no** new DB tables/migrations.

## Read path and traceability

- Uses the **same Postgres read path** as `app.retrieval.cli`: pin or auto-resolve one `IndexManifestRecord` via `resolve_manifest_record`, then `RetrievalOrchestrator` with `DenseRetriever` + `SparseRetriever`.
- **No new `processing_runs` rows** are created for generation (run_type `generation` remains unused; Phase 5/5.5 read-only invariant extends here). Ingest / chunk / index remain the write-side traces.
- `GroundedAnswer.metadata` echoes `index_manifest_id`, `manifest_hash`, embedding family, retrieval mode, context builder counts, and prompt/answer char counts for reproducibility.

## Context builder

`ContextBuilder` is **pure**: no SQLAlchemy, no LLM. It:

- preserves retrieval order;
- assigns `citation_id` from 1 upward;
- applies `max_chunks`, `max_context_chars`, `max_chunk_chars`, optional `min_score` (uses branch score: RRF > dense > sparse);
- truncates long chunk text with `…` (U+2026);
- copies `chunk_id`, `document_id`, `source_path`, `title`, `heading`, `rank_position`, and score.

## Prompt format

`build_grounded_prompt` emits a deterministic instruction block followed by numbered context headers:

```text
[N] source: <path-or—> | title: <title-or—> | heading: <heading-or—>
<text line(s)>
```

Missing optional fields render as **em dash (`—`)** so each header stays on one line.

If there are zero context blocks (e.g. retrieval returned no hits usable after filters), the prompt still builds with `Context:` → `(none)`; the orchestration short-circuits to `insufficient_context` **without** calling the provider when retrieval produces no blocks after `ContextBuilder`.

## Generation providers

Shipped Phase 6:

- **`GenerationProvider` protocol** (`app/generation/providers/base.py`): `name: str` and `generate(self, prompt: str) -> str`.
- **`INSUFFICIENT_CONTEXT_SENTENCE`**: exact fallback string the prompt instructs the model to use when context is inadequate.
- **`MockGenerationProvider`** (`mock`): parses `[N]` **block headers** in the prompt and returns a fixed English sentence citing up to the first three ids; if no headers are found, returns the insufficient-context sentence exactly.

CLI flag `--provider` only accepts **`mock`** in this phase (default follows `GENERATION_PROVIDER` env; invalid values exit before DB access).

Future (out of Phase 6): local / cloud LLMs behind optional adapters only.

## Answer modes

- **`grounded`**: extracted `[k]` cites at least one `k` matching a supplied context block.
- **`partial`**: context existed but raw answer cites nothing valid (or only out-of-range ids).
- **`insufficient_context`**: no retrieval-backed context survived `ContextBuilder`, or raw answer trimmed equals `INSUFFICIENT_CONTEXT_SENTENCE`.

`GroundedCitation` rows list **every** retained context block when not insufficient; `used_citation_ids` lists only ids present in raw answer brackets.

## Citation helpers

`extract_cited_ids` extracts digit-only `[N]` occurrences in appearance order with dedupe. **`verify_citations`** remains `NotImplementedError` (Phase 7: overlap / claim alignment).

## Operator CLI

Requires migrated Postgres + prior ingest/chunk/index (same prerequisites as retrieval).

```bash
python -m app.generation.cli "What does the basic intro file describe?" \
  --mode hybrid \
  --chunking-strategy fixed_size \
  --top-k 5 \
  --provider mock \
  --json
```

Options reuse Phase 5 manifest filters (`--index-manifest-id`, `--embedding-provider`, `--embedding-*`, `--chunking-strategy`, `--dense-top-k`, `--sparse-top-k`, `--rrf-k`). Context sizing: `--max-context-chars`, `--max-chunks`, `--max-chunk-chars`, `--min-score`.

## Programmatic API

Use `GroundedAnswerer.from_session(session, RetrievalConfig(...))` inside your own SQLAlchemy `session_scope` (same pattern as retrieval/evaluation runners).

## Hybrid prerequisite

Hybrid retrieval/generation requires a manifest with **`include_sparse=true`** — same validation as Phase 5 / 5.5 hybrid paths.

## Out of scope (Phase 6)

- FastAPI `/v1/ask` or auth.
- Streaming, async generators, speculative decoding.
- Reranking (`app/retrieval/rerank.py` stays placeholder).
- Answer-quality benchmarks or LLM-as-judge evaluation.
- Advanced citation verification (Phase 7).
- Mandatory paid API keys.

## Tests

- Default `pytest`: unit tests without DB (`tests/unit/test_generation_*.py`).
- Integration smoke: `LEGAL_RAG_RUN_INTEGRATION_DB=1` then `pytest -m integration tests/integration/test_generation_persist.py`.
