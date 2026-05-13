"""Optional persisted chunking tests (Postgres + LEGAL_RAG_RUN_INTEGRATION_DB=1)."""

from __future__ import annotations

import os

import pytest
from app.chunking.models import ChunkingConfig
from app.chunking.runner import chunk_document_persisted
from app.ingestion.runner import ingest_file_persisted
from app.ingestion.services import default_ingestion_service
from app.storage.postgres.session import get_engine, invalidate_engine_cache
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

RUN_DB_INTEGRATION = os.environ.get("LEGAL_RAG_RUN_INTEGRATION_DB") == "1"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not RUN_DB_INTEGRATION, reason="Set LEGAL_RAG_RUN_INTEGRATION_DB=1 to run")
def test_chunk_document_persisted_idempotent_and_conflict(tmp_path) -> None:
    invalidate_engine_cache()
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")

    path = tmp_path / "chunk_me.md"
    path.write_text("# Title\n\n" + ("body line\n" * 50), encoding="utf-8")
    svc = default_ingestion_service()
    ing = ingest_file_persisted(path, svc)
    assert ing.error is None
    assert ing.document is not None
    doc_id = ing.document.id
    assert doc_id is not None

    cfg_a = ChunkingConfig(strategy="fixed_size", chunk_size=400, chunk_overlap=40)
    r1 = chunk_document_persisted(doc_id, cfg_a)
    assert r1.error is None
    assert r1.created_chunks >= 1
    assert r1.skipped_existing is False

    r2 = chunk_document_persisted(doc_id, cfg_a)
    assert r2.error is None
    assert r2.skipped_existing is True

    cfg_b = ChunkingConfig(strategy="fixed_size", chunk_size=120, chunk_overlap=10)
    r3 = chunk_document_persisted(doc_id, cfg_b)
    assert r3.error is not None
    assert r3.created_chunks == 0
