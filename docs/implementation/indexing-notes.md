# Indexing notes — Phase 4A

## Goal

Add **indexing foundations** between persisted chunks and retrieval:

- Explicit `EmbeddingProvider` protocol + **deterministic hash** pseudo-vectors for tests/dev (no downloads, no external APIs).
- Sparse representation as **term → count** maps (tokenizer + extractor), **not** Postgres FTS yet.
- Persist **`index_manifests`** + **`index_manifest_chunks`** for reproducibility and audit.
- Operator entry: **`python -m app.indexing.cli`** (same pattern as ingestion/chunking CLIs).

This phase validates **interfaces and orchestration**. It does **not**:

- Store real vectors in `chunks` / pgvector columns (defer to Phase 4B once model dimensions are fixed).
- Create `tsvector`/GIN or hybrid retrieval (Phase 5+).
- Add `/v1/*` routes or Streamlit wiring.

## Deterministic embeddings

`DeterministicHashEmbeddingProvider` derives fixed-size float vectors from SHA-256 of UTF-8 text blocks.

**Not semantic retrieval quality.** Use only for scaffolding, CI, and smoke indexing without GPU/models.

Real **local embeddings** → Phase 4B (`sentence-transformers` or similar behind the same protocol).

## Sparse terms

`extract_sparse_terms` returns lowercase alphanumeric token frequencies. No stemming/stopwords in Phase 4A unless documented later.

## Manifests and idempotency

Hashes:

- **`config_hash`**: canonical JSON of indexing options (excluding `force_reindex`).
- **`chunk_set_hash`**: fingerprints of selected chunk ids + strategy + checksum (order-independent set hash).
- **`manifest_hash`**: `SHA256(config_hash + ":" + chunk_set_hash)`.

If **`manifest_hash`** already exists and **`force_reindex`** is false, the runner completes with **`skipped_existing`** on a new `processing_runs` row pointing at the existing manifest (`existing_manifest_id` metadata).

Changing chunks (new rows, edits, checksum changes) → new **`chunk_set_hash`** → no false skip.

## Persistence

Tables: **`index_manifests`**, **`index_manifest_chunks`**. No vector column on **`chunks`** in Phase 4A.

## Commands

```bash
# After migrate + ingest + chunk (see README)
python -m app.indexing.cli --chunking-strategy fixed_size --json
python -m app.indexing.cli --json   # all strategies
```

## Tests

Default `pytest` is DB-free. Integration: `LEGAL_RAG_RUN_INTEGRATION_DB=1`, `pytest -m integration`.

## Next phases

| Phase | Dense / storage | Sparse / retrieval |
|-------|-----------------|-------------------|
| **4B** | Real local embeddings + pgvector column + index | Optional `tsvector`/GIN |
| **5** | Dense retrieval | Sparse + hybrid + RRF |
