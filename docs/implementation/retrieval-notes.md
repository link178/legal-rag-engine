# Retrieval notes — Phase 5

## Goal

Phase 5 adds **read-side retrieval** over persisted chunks: dense search via **pgvector** (`chunk_embeddings`), a **baseline lexical** sparse branch over **`index_manifest_chunks.sparse_terms_json`**, optional **RRF** fusion for hybrid mode, and an operator **CLI**. There is **no** answer generation, no `/v1/ask`, and **no** new `processing_runs` rows for retrieval (ingest/chunk/index remain the write-side traces).

## Dense retrieval

- Vectors are stored per `(chunk_id, index_manifest_id)` in `chunk_embeddings`.
- Queries are embedded with the **same** provider family as the manifest (`build_embedding_provider` + `IndexingConfig` derived from the manifest row).
- Similarity uses **L2 distance** (`<->`) on pgvector; displayed **dense_score** is `1 / (1 + distance)` (higher is better). Raw distance is copied into `RetrievedChunk.metadata["dense_distance"]`.
- Each query pins **one** manifest (explicit `--index-manifest-id` or auto-select latest with `embeddings_persisted=true` and optional family filters).

## Sparse retrieval (baseline)

- Term frequencies come **only** from `index_manifest_chunks.sparse_terms_json` for chunks with `sparse_indexed=true` (same tokenizer as indexing: `app/indexing/sparse/tokenizer.py`).
- Scoring is a **BM25-lite** IDF over the manifest’s sparse chunk set, plus a sublinear TF term. This is **not** Postgres FTS and not full BM25 with length normalization.
- Auto-select for sparse-only mode uses the latest manifest with `include_sparse=true` (optional filters). It does **not** require `embeddings_persisted`.

## Hybrid and RRF

- **Hybrid** requires a manifest with **`include_sparse=true`** and a dense path (persisted embeddings).
- **RRF**: `score += 1 / (rrf_k + rank)` per list (rank 1-based), merged by `chunk_id`, tie-break `(rrf_score desc, chunk_id asc)`.
- If one branch returns no hits, fusion still runs on the non-empty list.

## Operator CLI

```bash
python -m app.retrieval.cli "your query" --json
python -m app.retrieval.cli "your query" --mode dense_only --top-k 5 --json
python -m app.retrieval.cli "your query" --mode sparse_only --chunking-strategy fixed_size --json
python -m app.retrieval.cli "your query" --mode hybrid --index-manifest-id <uuid> --json
```

Defaults: `RETRIEVAL_MODE` for `--mode`; embedding family filters default from `EMBEDDING_*` in settings when auto-selecting a manifest.

Prerequisites: migrated Postgres, ingest + chunk + `python -m app.indexing.cli`.

## Metadata-aware retrieval filters (Phase 14)

- Optional **exact-match AND** constraints on scalar fields in `documents.metadata_json` (e.g. `jurisdiction`, `legal_document_type`, `corpus_name`). Implemented in repositories as JSONB `.contains({key: value})` per field; **empty / omitted filter** produces the same SQL as before Phase 14.
- Wired through `RetrievalConfig.metadata_filter`, retrieval / answer HTTP bodies (`metadata_filter`), operator CLIs (`--filter-*` via `app/retrieval/filter_cli.py`), evaluation CLIs, and echoed in `RetrievalResultSet.metadata`, `GroundedAnswer.metadata`, and evaluation `summary.config`.
- **Sparse / IDF:** when a filter is applied, BM25-lite statistics (`N`, document frequencies) are computed only over chunks whose documents pass the filter (IDF is consistent in that subspace).
- **Not supported:** OR/IN/ranges/regex, nested operators, or per-question `metadata_filter` inside golden JSONL (deferred).

## Out of scope (Phase 5)

- pgvector **ANN** indexes (HNSW/IVFFlat).
- Postgres **FTS** / `tsvector`.
- **Reranking** (`app/retrieval/rerank.py` remains a placeholder).
- **Generation**, citations plumbing, Streamlit, external APIs (**Phase 6 generation CLI + mock provider shipped** — see [`generation-notes.md`](generation-notes.md); still no `/v1/ask`).

## Tests

- Default `pytest` is DB-free; retrieval logic is covered with mocks.
- Integration: `LEGAL_RAG_RUN_INTEGRATION_DB=1` and `pytest -m integration` (see `tests/integration/test_retrieval_persist.py`, `tests/integration/test_retrieval_filter_persist.py`).
- Retrieval-only evaluation (Phase 5.5): `app.evaluation.cli` — see [`evaluation-notes.md`](evaluation-notes.md).
