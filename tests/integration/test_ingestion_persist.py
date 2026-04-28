"""Optional: persisted ingest requires Postgres, migrations, and LEGAL_RAG_RUN_INTEGRATION_DB=1."""

from __future__ import annotations

import os

import pytest
from app.ingestion.runner import ingest_file_persisted
from app.ingestion.services import default_ingestion_service
from app.storage.postgres.session import get_engine, invalidate_engine_cache
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

RUN_DB_INTEGRATION = os.environ.get("LEGAL_RAG_RUN_INTEGRATION_DB") == "1"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not RUN_DB_INTEGRATION, reason="Set LEGAL_RAG_RUN_INTEGRATION_DB=1 to run")
def test_ingest_file_persisted_round_trip(tmp_path) -> None:
    invalidate_engine_cache()
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")

    p = tmp_path / "persist.txt"
    p.write_text("persist me\n", encoding="utf-8")
    svc = default_ingestion_service()
    r1 = ingest_file_persisted(p, svc)
    assert r1.error is None
    assert r1.created is True
    assert r1.document is not None
    assert r1.document.created_by_run_id == r1.run.id

    r2 = ingest_file_persisted(p, svc)
    assert r2.error is None
    assert r2.skipped is True
    assert r2.created is False
