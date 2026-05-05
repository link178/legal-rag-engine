"""Optional answer evaluation over Postgres + LEGAL_RAG_RUN_INTEGRATION_DB=1."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from app.chunking.models import ChunkingConfig
from app.chunking.runner import chunk_document_persisted
from app.core.config import get_settings
from app.evaluation.runners.answer import AnswerEvaluationRunner
from app.indexing.models import IndexingConfig
from app.indexing.runner import index_chunks_persisted
from app.ingestion.runner import ingest_file_persisted
from app.ingestion.services import default_ingestion_service
from app.retrieval.models import RetrievalConfig
from app.storage.postgres.models import ProcessingRunRecord
from app.storage.postgres.session import get_engine, invalidate_engine_cache, session_scope
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError

RUN_DB_INTEGRATION = os.environ.get("LEGAL_RAG_RUN_INTEGRATION_DB") == "1"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not RUN_DB_INTEGRATION, reason="Set LEGAL_RAG_RUN_INTEGRATION_DB=1 to run")
def test_answer_evaluation_wiring_sample_corpus() -> None:
    invalidate_engine_cache()
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")

    with session_scope() as session:
        n_eval_runs = session.scalar(
            select(func.count()).select_from(ProcessingRunRecord).where(
                ProcessingRunRecord.run_type == "evaluation"
            )
        )
    assert n_eval_runs == 0

    root = Path(__file__).resolve().parents[2]
    intro = root / "data/sample_corpus/basic/intro.md"
    plain = root / "data/sample_corpus/basic/plain.txt"
    golden = root / "data/eval/answer_golden.jsonl"

    svc = default_ingestion_service()
    ing1 = ingest_file_persisted(intro, svc)
    ing2 = ingest_file_persisted(plain, svc)
    assert ing1.error is None and ing1.document and ing1.document.id
    assert ing2.error is None and ing2.document and ing2.document.id
    doc1 = ing1.document.id
    doc2 = ing2.document.id
    assert doc1 and doc2

    cfg_ck = ChunkingConfig(strategy="fixed_size", chunk_size=400, chunk_overlap=40)
    c1 = chunk_document_persisted(doc1, cfg_ck)
    c2 = chunk_document_persisted(doc2, cfg_ck)
    assert c1.error is None
    assert c2.error is None

    ix_cfg = IndexingConfig(chunking_strategy="fixed_size", embedding_dimensions=16, batch_size=8)
    r1 = index_chunks_persisted(ix_cfg)
    assert r1.error is None
    assert r1.manifest is not None
    mid = r1.manifest.id
    assert mid is not None

    settings = get_settings()
    cfg = RetrievalConfig(
        mode="sparse_only",
        chunking_strategy="fixed_size",
        top_k=5,
        index_manifest_id=mid,
    )

    with session_scope(settings.database_url) as session:
        runner = AnswerEvaluationRunner.from_session(session, cfg)
        summary = runner.run_file(golden, top_k=5)

    assert summary.total_questions == 3
    assert summary.total_questions == summary.answered_questions + summary.errored_questions
    assert summary.schema_version == "evaluation.answer.v1"
    assert summary.manifest.get("id")
    assert summary.errored_questions == 0

    with session_scope() as session:
        n_after = session.scalar(
            select(func.count()).select_from(ProcessingRunRecord).where(
                ProcessingRunRecord.run_type == "evaluation"
            )
        )
    assert n_after == 0
