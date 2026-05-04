"""Optional end-to-end retrieval tests (Postgres + LEGAL_RAG_RUN_INTEGRATION_DB=1)."""

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
from app.retrieval.dense import DenseRetriever
from app.retrieval.manifest import resolve_manifest_record
from app.retrieval.models import RetrievalConfig
from app.retrieval.orchestrator import RetrievalOrchestrator
from app.retrieval.sparse import SparseRetriever
from app.storage.postgres.models import ProcessingRunRecord
from app.storage.postgres.session import get_engine, invalidate_engine_cache, session_scope
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError

RUN_DB_INTEGRATION = os.environ.get("LEGAL_RAG_RUN_INTEGRATION_DB") == "1"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not RUN_DB_INTEGRATION, reason="Set LEGAL_RAG_RUN_INTEGRATION_DB=1 to run")
def test_retrieval_dense_sparse_hybrid_idempotent(tmp_path: Path) -> None:
    invalidate_engine_cache()
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")

    with session_scope() as session:
        n_retrieval_runs = session.scalar(
            select(func.count()).select_from(ProcessingRunRecord).where(
                ProcessingRunRecord.run_type == "retrieval"
            )
        )
    assert n_retrieval_runs == 0

    p = tmp_path / "retrieve_me.md"
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
    assert r1.manifest is not None
    mid = r1.manifest.id
    assert mid is not None

    with session_scope() as session:
        cfg_d = RetrievalConfig(
            mode="dense_only",
            chunking_strategy="fixed_size",
            top_k=5,
            index_manifest_id=mid,
        )
        mf = resolve_manifest_record(session, cfg_d, for_dense=True)
        orch = RetrievalOrchestrator(DenseRetriever(session), SparseRetriever(session))
        dres = orch.retrieve("paragraph", cfg_d, mf)

    assert dres.results
    assert any(x.document_id == doc_id for x in dres.results)
    assert all(x.dense_score is not None and x.dense_score > 0 for x in dres.results)
    assert all(x.rrf_score is None for x in dres.results)

    with session_scope() as session:
        cfg_s = RetrievalConfig(
            mode="sparse_only",
            chunking_strategy="fixed_size",
            top_k=5,
            index_manifest_id=mid,
        )
        mf_s = resolve_manifest_record(session, cfg_s, for_dense=False)
        orch_s = RetrievalOrchestrator(DenseRetriever(session), SparseRetriever(session))
        sres = orch_s.retrieve("paragraph", cfg_s, mf_s)

    assert sres.results
    assert all(x.sparse_score is not None and x.sparse_score >= 0 for x in sres.results)
    assert all(x.rrf_score is None for x in sres.results)

    with session_scope() as session:
        cfg_h = RetrievalConfig(
            mode="hybrid",
            chunking_strategy="fixed_size",
            top_k=5,
            index_manifest_id=mid,
        )
        mf_h = resolve_manifest_record(session, cfg_h, for_dense=True)
        assert mf_h.include_sparse is True
        orch_h = RetrievalOrchestrator(DenseRetriever(session), SparseRetriever(session))
        h1 = orch_h.retrieve("paragraph", cfg_h, mf_h)

    assert h1.results
    assert all(x.rrf_score is not None for x in h1.results)
    for x in h1.results:
        assert set(x.retrieval_sources).issubset({"dense", "sparse"})
        assert len(x.retrieval_sources) >= 1

    with session_scope() as session:
        cfg_h2 = RetrievalConfig(
            mode="hybrid",
            chunking_strategy="fixed_size",
            top_k=5,
            index_manifest_id=mid,
        )
        mf_h2 = resolve_manifest_record(session, cfg_h2, for_dense=True)
        orch_h2 = RetrievalOrchestrator(DenseRetriever(session), SparseRetriever(session))
        h2 = orch_h2.retrieve("paragraph", cfg_h2, mf_h2)

    assert [x.chunk_id for x in h1.results] == [x.chunk_id for x in h2.results]

    with session_scope() as session:
        n_after = session.scalar(
            select(func.count()).select_from(ProcessingRunRecord).where(
                ProcessingRunRecord.run_type == "retrieval"
            )
        )
    assert n_after == 0
