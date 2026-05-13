# Indexing notes — Phases 4A & 4B

## Phase 4A — Foundations

Add **indexing foundations** between persisted chunks and retrieval:

- Explicit `EmbeddingProvider` protocol + **deterministic hash** pseudo-vectors for tests/dev (no downloads, no external APIs).
- Sparse representation as **term → count** maps (tokenizer + extractor), **not** Postgres FTS yet.
- Persist **`index_manifests`** + **`index_manifest_chunks`** for reproducibility and audit.
- Operator entry: **`python -m app.indexing.cli`** (same pattern as ingestion/chunking CLIs).

Phase 4A alone does **not** store real vectors in Postgres.

## Phase 4B — pgvector storage + optional local provider

Phase 4B adds:

- Table **`chunk_embeddings`**: one row per `( chunk_id, index_manifest_id )` with a **pgvector** `embedding` column (no fixed dimension in v1 storage; optional **ANN** indexes deferred to Phase 5).
- Manifest columns **`embedding_model`** and **`embeddings_persisted`**: skip (`skipped_existing`) only when the latest manifest for the same `manifest_hash` has **`embeddings_persisted = true`** and `force_reindex` is false. Older 4A-only rows have `embeddings_persisted = false` and will be re-built (new manifest row; same or new hash depending on config keys).
- **`IndexingConfig.embedding_model`** participates in **`config_hash`** so different HF models never share the same idempotent key by accident.
- Optional provider **`local_sentence_transformers`**: `SentenceTransformersEmbeddingProvider` in `app/indexing/dense/local_provider.py` (lazy import / lazy model load). Install: `pip install -e ".[local-embeddings]"`. **`embedding_dimensions`** must match the model output size (e.g. 384 for `intfloat/multilingual-e5-small`).
- Factory: **`app/indexing.dense.factory.build_embedding_provider`**.

Still **not** in Phase 4B:

- Similarity search / top-k queries against pgvector.
- Dense or hybrid **retrieval**, RRF, reranking.
- `tsvector` / GIN (optional follow-up).
- `/v1/*` routes or Streamlit.
- External embedding APIs; LangChain / LlamaIndex; Qdrant.

## Deterministic embeddings

`DeterministicHashEmbeddingProvider` derives fixed-size float vectors from SHA-256 of UTF-8 text blocks.

**Not semantic retrieval quality.** Use only for scaffolding, CI, and smoke indexing without GPU/models.

## Sparse terms

`extract_sparse_terms` returns lowercase alphanumeric token frequencies. No stemming/stopwords unless documented later.

## Manifests and idempotency

Hashes:

- **`config_hash`**: canonical JSON of indexing options (excluding `force_reindex`), including **`embedding_model`**.
- **`chunk_set_hash`**: fingerprints of selected chunk ids + strategy + checksum (order-independent set hash).
- **`manifest_hash`**: `SHA256(config_hash + ":" + chunk_set_hash)`.

If the latest manifest for **`manifest_hash`** has **`embeddings_persisted`** and **`force_reindex`** is false, the runner completes with **`skipped_existing`**. Otherwise a new manifest row is created (and new **`chunk_embeddings`** rows when dense is enabled).

## Persistence (summary)

| Table | Phase | Role |
|-------|-------|------|
| `index_manifests` | 4A+ | Build snapshot, hashes, provider/dims/model, **`embeddings_persisted`**. |
| `index_manifest_chunks` | 4A+ | Per-chunk dense/sparse flags + optional **`sparse_terms_json`**. |
| `chunk_embeddings` | 4B | Dense vector + checksum + provider/model FK’d to manifest and chunk. |

No embedding column on **`chunks`**.

## Commands

```bash
# After migrate + ingest + chunk (see README)
python -m app.indexing.cli --chunking-strategy fixed_size --json

# Explicit provider/dims (defaults also from EMBEDDING_* env vars)
python -m app.indexing.cli --chunking-strategy fixed_size \
  --embedding-provider deterministic_hash --embedding-dimensions 16 --json

# Optional real local model (requires `.[local-embeddings]`; no auto-download in tests)
python -m app.indexing.cli --chunking-strategy fixed_size \
  --embedding-provider local_sentence_transformers \
  --embedding-model intfloat/multilingual-e5-small --embedding-dimensions 384 --json
```

## Tests

Default `pytest` is DB-free. Integration: `LEGAL_RAG_RUN_INTEGRATION_DB=1`, `pytest -m integration`.

## Next phases

| Phase | Dense / storage | Sparse / retrieval |
|-------|-----------------|-------------------|
| **5** (done in repo) | Dense retrieval over `chunk_embeddings` (no ANN index yet) | Baseline lexical over `sparse_terms_json` + hybrid RRF (see [`retrieval-notes.md`](retrieval-notes.md)) |
| **6+** | Optional pgvector ANN | Postgres FTS / GIN optional |
