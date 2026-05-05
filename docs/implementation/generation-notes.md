# Generation notes — Phases 6–7 (mock grounded QA + citation verification)

## Goal

Phases 6–7 wire **operator-controlled grounded answering** after retrieval with **mechanical citation verification**:

```text
question
  → RetrievalOrchestrator.retrieve (same contract as Phase 5)
  → ContextBuilder (ranked snippets → citation blocks [1],[2],…)
  → build_grounded_prompt(full prompt string)
  → GenerationProvider.generate(prompt) → raw answer text
  → verify_citations(raw answer, context blocks) [Phase 7]
  → GroundedAnswer (mode + citations + citation_verification + metadata)
```

There is **no** `POST /v1/ask`, **no** OpenAI/Ollama, **no** streaming, **no** new DB tables/migrations.

## Read path and traceability

- Uses the **same Postgres read path** as `app.retrieval.cli`: pin or auto-resolve one `IndexManifestRecord` via `resolve_manifest_record`, then `RetrievalOrchestrator` with `DenseRetriever` + `SparseRetriever`.
- **No new `processing_runs` rows** are created for generation (run_type `generation` remains unused; Phase 5/5.5 read-only invariant extends here). Ingest / chunk / index remain the write-side traces.
- `GroundedAnswer.metadata` echoes `index_manifest_id`, `manifest_hash`, embedding family, retrieval mode, context builder counts, prompt/answer char counts, plus `citation_validity_rate` and `has_invalid_citations` for operator traceability.
- **Phase 14:** when `RetrievalConfig.metadata_filter` is non-empty, `GroundedAnswer.metadata` also includes `metadata_filter` (same key/value map as retrieval). Generation CLI accepts the shared `--filter-*` flags (`app/retrieval/filter_cli.py`).

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

- **`grounded`**: at least one bracketed `[k]` refers to a context block **and** **no** bracketed id is out of range (all used unique ids are valid).
- **`partial`**: context existed but the answer has no valid brackets, only invalid/out-of-range brackets, or **mixed** valid + invalid brackets.
- **`insufficient_context`**: no retrieval-backed context survived `ContextBuilder`, or raw answer trimmed equals `INSUFFICIENT_CONTEXT_SENTENCE`.

`GroundedCitation` rows list **every** retained context block when not insufficient; `used_citation_ids` lists **valid** cited ids only (subset of context block ids). Full mechanical detail is on `GroundedAnswer.citation_verification`.

## Citation verification (Phase 7)

Phase 6 instructed models to cite with `[N]`; Phase 7 **verifies mechanically** that those ids exist in the **same** `GroundedContextBlock` list passed to the prompt:

- **Citation extraction**: `extract_cited_ids` returns unique ids in **first-appearance order** (deduped) for stable `used_citation_ids`-style consumers.
- **Citation verification**: `verify_citations(answer_text, context_blocks)` scans **every** `[N]` occurrence (including duplicates), computes `valid_*` / `invalid_*` / `unused_*` / `duplicate_*`, and `citation_validity_rate = |valid unique| / |used unique|` (0.0 when nothing cited).
- **Not** claim-level or semantic support: it does not prove each sentence is entailed by cited chunks; that remains future work.

`GroundedAnswerer` always attaches a `CitationVerificationResult` (vacuous when `insufficient_context`, with `available_citation_ids` filled when the provider returned the sentinel but blocks had existed).

CLI `--json` maps this under `citation_verification`; human output prints **Citation validity**, **Valid citations**, **Invalid citations**, and **Unused citations**.

## Citation helpers (legacy summary)

`extract_cited_ids` — deduped first-appearance order. `verify_citations` — full mechanical report (see above).

Example `citation_verification` in JSON (CLI / `answer_to_dict`):

```json
{
  "used_citation_ids": [1],
  "available_citation_ids": [1, 2],
  "valid_citation_ids": [1],
  "invalid_citation_ids": [],
  "unused_citation_ids": [2],
  "duplicate_citation_ids": [],
  "citation_validity_rate": 1.0,
  "has_citations": true,
  "has_valid_citations": true,
  "has_invalid_citations": false
}
```

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

## Out of scope (Phases 6–7)

- FastAPI `POST /v1/answer` (Phase 8) — see [`api-notes.md`](api-notes.md); still no `POST /v1/ask`.
- Streaming, async generators, speculative decoding.
- Reranking (`app/retrieval/rerank.py` stays placeholder).
- Answer-quality benchmarks or LLM-as-judge evaluation.
- Semantic / claim-by-claim citation or answer verification.
- Mandatory paid API keys.

## Tests

- Default `pytest`: unit tests without DB (`tests/unit/test_generation_*.py`).
- Integration smoke: `LEGAL_RAG_RUN_INTEGRATION_DB=1` then `pytest -m integration tests/integration/test_generation_persist.py`.
