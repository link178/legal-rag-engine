# Portfolio summary — legal-rag-engine

## One-liner

Local-first hybrid RAG engine with PostgreSQL/pgvector, metadata-aware retrieval, mechanical citation verification, and reproducible retrieval and answer evaluation.

## What it is

An **engine-first** Python codebase—not a chatbot wrapper—covering traceable ingestion (Markdown, text, HTML, PDF text extraction), chunking strategies, dense and sparse retrieval with **RRF** fusion, indexing manifests, grounded **mock** generation with citations, a thin FastAPI `/v1` surface, and a Streamlit demo that calls the API over HTTP only.

## What it demonstrates

- End-to-end RAG plumbing with **operator-controlled** CLIs and optional HTTP.
- **Deterministic baselines**: pseudo-embeddings and mock generation so tests and demos run without paid APIs.
- **Evaluation discipline**: golden JSONL for retrieval (Hit@k, MRR) and for grounded answers (mock end-to-end checks).
- **Honest boundaries**: not legal advice, not a compliance SaaS, not an AI Act product.

## Technical highlights

- PostgreSQL + **pgvector** as the default vector store.
- Retrieval modes **`dense_only`**, **`sparse_only`**, **`hybrid`** (explicit naming aligned with the API).
- Corpus adapter for **`legalize-*`** style trees with enriched metadata and import reports.
- Seven-field **exact-match metadata filters** on document JSON (retrieval, generation, evaluation, and Streamlit).

## What I learned / demonstrated

- Separating **engine** (domain, runners, repositories) from **delivery** (FastAPI, Streamlit).
- Treating **index manifests** as the unit of reproducibility for “what was indexed.”
- Keeping **citation verification** mechanical and testable rather than claiming entailment.

## Limitations (fair)

- Default generation is **mock**; real LLM integration is explicitly out of scope for v0.1.0.
- PDFs: text extraction only; **no OCR**.
- Sparse retrieval is a **manifest-bound** lexical baseline in v1, not a full FTS product.
- The project targets **portfolio-grade engineering**, not production SaaS operations (no auth, no multi-tenant layer in-repo).
