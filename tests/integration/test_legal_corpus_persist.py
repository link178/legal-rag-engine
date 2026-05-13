"""Optional: persisted legal corpus import requires Postgres + LEGAL_RAG_RUN_INTEGRATION_DB=1."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

import pytest
from app.ingestion.adapters.legal_corpus.importer import LegalCorpusImporter
from app.ingestion.services import default_ingestion_service
from app.storage.postgres.models import ProcessingRunRecord
from app.storage.postgres.session import get_engine, invalidate_engine_cache, session_scope
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

RUN_DB_INTEGRATION = os.environ.get("LEGAL_RAG_RUN_INTEGRATION_DB") == "1"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not RUN_DB_INTEGRATION, reason="Set LEGAL_RAG_RUN_INTEGRATION_DB=1 to run")
def test_legal_corpus_persist_umbrella_and_idempotent() -> None:
    invalidate_engine_cache()
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")

    repo_root = Path(__file__).resolve().parents[2]
    sample = repo_root / "data" / "sample_corpus" / "legalize_sample"

    svc = default_ingestion_service()
    importer = LegalCorpusImporter(svc)

    summary1 = importer.import_corpus(sample, persist=True)
    assert summary1.discovered_count == 3
    assert summary1.imported_count == 3
    assert summary1.reused_count == 0
    assert summary1.failed_count == 0
    assert summary1.corpus_run_id is not None

    umbrella_uuid = UUID(summary1.corpus_run_id)

    with session_scope() as session:
        umbrella = session.get(ProcessingRunRecord, umbrella_uuid)
        assert umbrella is not None
        assert umbrella.run_type == "corpus_import"
        assert umbrella.status == "completed"

        stmt = select(ProcessingRunRecord).where(ProcessingRunRecord.run_type == "ingest")
        ingest_rows = list(session.scalars(stmt).all())
        linked = [
            r
            for r in ingest_rows
            if r.metadata_json.get("corpus_run_id") == summary1.corpus_run_id
        ]
        assert len(linked) == 3

    summary2 = importer.import_corpus(sample, persist=True)
    assert summary2.imported_count == 0
    assert summary2.reused_count == 3
    assert summary2.failed_count == 0
