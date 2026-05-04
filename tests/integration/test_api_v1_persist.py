"""Optional API v1 + DB flow (Postgres + LEGAL_RAG_RUN_INTEGRATION_DB=1)."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

import pytest
from app.main import app
from app.storage.postgres.models import ProcessingRunRecord
from app.storage.postgres.session import get_engine, invalidate_engine_cache, session_scope
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError

RUN_DB_INTEGRATION = os.environ.get("LEGAL_RAG_RUN_INTEGRATION_DB") == "1"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not RUN_DB_INTEGRATION, reason="Set LEGAL_RAG_RUN_INTEGRATION_DB=1 to run")
def test_api_v1_ingest_retrieve_answer_no_extra_runs() -> None:
    invalidate_engine_cache()
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")

    with session_scope() as session:
        n_rx = session.scalar(
            select(func.count()).select_from(ProcessingRunRecord).where(
                ProcessingRunRecord.run_type.in_(("retrieval", "generation"))
            )
        )
    assert n_rx == 0

    root = Path(__file__).resolve().parents[2]
    intro = root / "data/sample_corpus/basic/intro.md"
    assert intro.is_file(), intro

    client = TestClient(app)
    ing = client.post("/v1/ingest", json={"path": str(intro), "persist": True})
    assert ing.status_code == 200, ing.text
    body = ing.json()
    assert body.get("error") is None
    doc_id_str = body.get("document_id")
    assert doc_id_str

    chk_resp = client.post(
        "/v1/chunk",
        json={
            "document_id": doc_id_str,
            "strategy": "fixed_size",
            "chunk_size": 400,
            "chunk_overlap": 40,
        },
    )
    assert chk_resp.status_code == 200, chk_resp.text
    chk_body = chk_resp.json()
    assert chk_body.get("chunks_count", 0) >= 1

    ix_resp = client.post(
        "/v1/index",
        json={
            "chunking_strategy": "fixed_size",
            "embedding_dimensions": 16,
            "batch_size": 8,
        },
    )
    assert ix_resp.status_code == 200, ix_resp.text
    ix_body = ix_resp.json()
    mid_str = ix_body.get("manifest_id")
    assert mid_str

    ml = client.get("/v1/index-manifests", params={"chunking_strategy": "fixed_size", "limit": 50})
    assert ml.status_code == 200, ml.text
    items = ml.json().get("manifests", [])
    ids = {m["manifest_id"] for m in items}
    assert mid_str in ids

    ix2 = client.post(
        "/v1/index",
        json={
            "chunking_strategy": "fixed_size",
            "embedding_dimensions": 16,
            "batch_size": 8,
        },
    )
    assert ix2.status_code == 200, ix2.text
    ix2_body = ix2.json()
    assert ix2_body.get("skipped_existing") is True
    assert ix2_body.get("manifest_id") == mid_str

    mid = UUID(mid_str)

    ret = client.post(
        "/v1/retrieve",
        json={
            "query": "What does the basic intro file describe?",
            "mode": "hybrid",
            "chunking_strategy": "fixed_size",
            "top_k": 5,
            "index_manifest_id": str(mid),
        },
    )
    assert ret.status_code == 200, ret.text
    rj = ret.json()
    assert rj.get("total_results", 0) >= 1
    assert rj.get("index_manifest_id") == str(mid)

    ans = client.post(
        "/v1/answer",
        json={
            "question": "What does the basic intro file describe?",
            "mode": "hybrid",
            "chunking_strategy": "fixed_size",
            "top_k": 5,
            "provider": "mock",
            "index_manifest_id": str(mid),
        },
    )
    assert ans.status_code == 200, ans.text
    aj = ans.json()
    assert aj["mode"] in ("grounded", "partial", "insufficient_context")
    if aj["mode"] == "grounded":
        assert len(aj.get("citations", [])) >= 1
        assert aj.get("used_citation_ids")

    with session_scope() as session:
        n_after = session.scalar(
            select(func.count()).select_from(ProcessingRunRecord).where(
                ProcessingRunRecord.run_type.in_(("retrieval", "generation"))
            )
        )
    assert n_after == 0
