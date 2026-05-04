"""Chunk embedding ORM table shape (metadata only)."""

from __future__ import annotations

from app.storage.postgres.models import ChunkEmbeddingRecord


def test_chunk_embeddings_table_name() -> None:
    assert ChunkEmbeddingRecord.__tablename__ == "chunk_embeddings"
