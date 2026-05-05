# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — Public portfolio release

### Added

- Local-first RAG engine baseline: FastAPI, PostgreSQL + pgvector, Alembic migrations.
- Document ingestion for Markdown, plain text, HTML, and PDF (text extraction; no OCR).
- Legal corpus adapter (`legalize-*`) with discovery, metadata enrichment, and umbrella `processing_runs`.
- Chunking strategies: fixed-size and Markdown structure-aware, with optional Postgres persistence.
- Dense (pgvector), sparse (manifest-bound lexical), and hybrid retrieval with RRF fusion.
- Indexing manifests and `chunk_embeddings` for traceability.
- Metadata-aware retrieval filters on `documents.metadata_json` (seven fields).
- Grounded mock generation, mechanical citation verification, insufficient-context handling.
- Retrieval evaluation (Hit@k / MRR) and answer evaluation (golden JSONL) via CLI.
- FastAPI API v1 (ingest, documents, chunk, index, manifests, retrieve, answer).
- Streamlit demo over the HTTP API (`app/ui/streamlit_app.py`).

### Not in 0.1.0

- Real LLM or cloud embedding providers as required dependencies (mock + deterministic_hash default).
- `POST /v1/ask`, multipart upload, auth, multi-tenant SaaS, compliance product scope.
- Semantic citation entailment, production rerankers, ANN indexes, PostgreSQL FTS as the sparse backend.
