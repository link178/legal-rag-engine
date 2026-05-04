"""Optional persisted indexing tests (Postgres + LEGAL_RAG_RUN_INTEGRATION_DB=1)."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
from app.chunking.models import ChunkingConfig
from app.chunking.runner import chunk_document_persisted
from app.indexing.models import IndexingConfig
from app.indexing.runner import index_chunks_persisted
from app.ingestion.runner import ingest_file_persisted
from app.ingestion.services import default_ingestion_service
from app.storage.postgres.models import ChunkEmbeddingRecord
from app.storage.postgres.repositories import ChunkEmbeddingRepository
from app.storage.postgres.session import get_engine, invalidate_engine_cache, session_scope
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError

RUN_DB_INTEGRATION = os.environ.get("LEGAL_RAG_RUN_INTEGRATION_DB") == "1"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not RUN_DB_INTEGRATION, reason="Set LEGAL_RAG_RUN_INTEGRATION_DB=1 to run")
def test_indexing_persisted_idempotent(tmp_path: Path) -> None:
    invalidate_engine_cache()
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")

    p = tmp_path / "index_me.md"
    p.write_text("# T\n\n" + ("paragraph\n" * 30), encoding="utf-8")
    svc = default_ingestion_service()
    ing = ingest_file_persisted(p, svc)
    assert ing.error is None and ing.document and ing.document.id
    doc_id = ing.document.id

    cfg_ck = ChunkingConfig(strategy="fixed_size", chunk_size=400, chunk_overlap=40)
    chk = chunk_document_persisted(doc_id, cfg_ck)
    assert chk.error is None

    ix_cfg = IndexingConfig(chunking_strategy="fixed_size", embedding_dimensions=16, batch_size=8)

    r1 = index_chunks_persisted(ix_cfg)
    assert r1.error is None
    assert r1.skipped_existing is False
    assert r1.manifest is not None
    assert r1.manifest.id is not None
    assert r1.manifest.embeddings_persisted is True

    mid1 = r1.manifest.id
    cc = r1.manifest.chunk_count

    with session_scope() as session:
        n_manifest = ChunkEmbeddingRepository(session).count_by_manifest(mid1)
        total_after_first = session.scalar(select(func.count(ChunkEmbeddingRecord.id)))
    assert n_manifest == cc
    assert total_after_first is not None and total_after_first >= cc

    r2 = index_chunks_persisted(ix_cfg)
    assert r2.error is None
    assert r2.skipped_existing is True
    assert r2.manifest is not None
    assert r2.manifest.id == mid1

    with session_scope() as session:
        n_after_skip = session.scalar(select(func.count(ChunkEmbeddingRecord.id)))
    assert n_after_skip == total_after_first

    r3 = index_chunks_persisted(replace(ix_cfg, force_reindex=True))
    assert r3.error is None
    assert r3.skipped_existing is False
    assert r3.manifest is not None
    mid3 = r3.manifest.id
    assert mid3 != mid1

    with session_scope() as session:
        er = ChunkEmbeddingRepository(session)
        assert er.count_by_manifest(mid1) == cc
        assert er.count_by_manifest(mid3) == cc
        total_final = session.scalar(select(func.count(ChunkEmbeddingRecord.id)))
    assert total_final == total_after_first + cc
