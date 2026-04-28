"""Optional persisted indexing tests (Postgres + LEGAL_RAG_RUN_INTEGRATION_DB=1)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from app.chunking.models import ChunkingConfig
from app.chunking.runner import chunk_document_persisted
from app.indexing.models import IndexingConfig
from app.indexing.runner import index_chunks_persisted
from app.ingestion.runner import ingest_file_persisted
from app.ingestion.services import default_ingestion_service
from app.storage.postgres.session import get_engine, invalidate_engine_cache
from sqlalchemy import text
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

    r2 = index_chunks_persisted(ix_cfg)
    assert r2.error is None
    assert r2.skipped_existing is True
