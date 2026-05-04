"""Inspect SQLAlchemy metadata without connecting to Postgres."""

from __future__ import annotations

from app.storage.postgres.models import (
    ChunkEmbeddingRecord,
    ChunkRecord,
    DocumentRecord,
    IndexManifestChunkRecord,
    IndexManifestRecord,
    ProcessingRunRecord,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import Mapper


def test_metadata_tables_exist() -> None:
    tables = {
        ProcessingRunRecord.__table__.name,
        DocumentRecord.__table__.name,
        ChunkRecord.__table__.name,
        IndexManifestRecord.__table__.name,
        IndexManifestChunkRecord.__table__.name,
        ChunkEmbeddingRecord.__table__.name,
    }
    assert tables == {
        "processing_runs",
        "documents",
        "chunks",
        "index_manifests",
        "index_manifest_chunks",
        "chunk_embeddings",
    }


def test_documents_columns() -> None:
    cols = {c.key for c in DocumentRecord.__table__.columns}
    assert cols == {
        "id",
        "source_path",
        "source_type",
        "title",
        "raw_text",
        "normalized_text",
        "metadata_json",
        "checksum",
        "created_by_run_id",
        "created_at",
        "updated_at",
    }


def test_chunks_columns_and_fk() -> None:
    cols = {c.key for c in ChunkRecord.__table__.columns}
    assert "text" in cols  # DB column name (ORM attribute is chunk_text)
    assert "document_id" in cols
    assert "created_by_run_id" in cols
    mapper = ChunkRecord.__mapper__
    assert isinstance(mapper, Mapper)
    fks = list(ChunkRecord.__table__.foreign_keys)
    assert len(fks) == 2
    fk_tables = {fk.column.table.name for fk in fks}
    assert fk_tables == {"documents", "processing_runs"}


def test_chunks_indexes_and_unique() -> None:
    index_names = {ix.name for ix in ChunkRecord.__table__.indexes if ix.name is not None}
    assert "ix_chunks_created_by_run_id" in index_names
    assert "ix_chunks_document_id" in index_names
    assert "ix_chunks_chunking_strategy" in index_names
    constraint_names = {c.name for c in ChunkRecord.__table__.constraints}
    assert "uq_chunks_document_chunk_strategy" in constraint_names


def test_documents_indexes_and_unique() -> None:
    constraint_names = {c.name for c in DocumentRecord.__table__.constraints}
    assert "uq_documents_source_path_checksum" in constraint_names


def test_index_manifests_columns() -> None:
    cols = {c.key for c in IndexManifestRecord.__table__.columns}
    assert cols == {
        "id",
        "run_id",
        "corpus_version",
        "chunking_strategy",
        "embedding_provider",
        "embedding_dimensions",
        "embedding_model",
        "embeddings_persisted",
        "document_count",
        "chunk_count",
        "indexed_chunk_count",
        "include_sparse",
        "include_dense",
        "config_hash",
        "chunk_set_hash",
        "manifest_hash",
        "metadata_json",
        "created_at",
        "updated_at",
    }


def test_index_manifest_chunks_fk() -> None:
    cols = {c.key for c in IndexManifestChunkRecord.__table__.columns}
    assert "manifest_id" in cols
    assert "chunk_id" in cols
    assert "sparse_terms_json" in cols
    fks = list(IndexManifestChunkRecord.__table__.foreign_keys)
    fk_tables = {fk.column.table.name for fk in fks}
    assert fk_tables == {"chunks", "index_manifests"}


def test_chunk_embeddings_columns_and_fk() -> None:
    cols = {c.key for c in ChunkEmbeddingRecord.__table__.columns}
    assert cols == {
        "id",
        "chunk_id",
        "index_manifest_id",
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "embedding",
        "text_checksum",
        "metadata_json",
        "created_at",
        "updated_at",
    }
    emb_col = ChunkEmbeddingRecord.__table__.c.embedding
    assert isinstance(emb_col.type, Vector)
    fks = list(ChunkEmbeddingRecord.__table__.foreign_keys)
    fk_tables = {fk.column.table.name for fk in fks}
    assert fk_tables == {"chunks", "index_manifests"}
    constraint_names = {c.name for c in ChunkEmbeddingRecord.__table__.constraints}
    assert "uq_chunk_embeddings_chunk_manifest" in constraint_names


def test_processing_runs_columns() -> None:
    cols = {c.key for c in ProcessingRunRecord.__table__.columns}
    assert "run_type" in cols
    assert "status" in cols
    assert "error_message" in cols
    assert "metadata_json" in cols
