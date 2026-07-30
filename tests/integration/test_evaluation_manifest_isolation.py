"""Explicit manifest_id isolates evaluation from unrelated newer manifests."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest
from app.chunking.models import ChunkingConfig
from app.chunking.runner import chunk_document_persisted
from app.core.config import get_settings
from app.evaluation.runners.answer import AnswerEvaluationRunner
from app.evaluation.runners.retrieval import RetrievalEvaluationRunner
from app.indexing.models import IndexingConfig
from app.indexing.runner import index_chunks_persisted
from app.ingestion.runner import ingest_file_persisted
from app.ingestion.services import default_ingestion_service
from app.retrieval.errors import ManifestNotFoundError
from app.retrieval.manifest import resolve_manifest_record
from app.retrieval.models import RetrievalConfig
from app.storage.postgres.session import get_engine, invalidate_engine_cache, session_scope
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

RUN_DB_INTEGRATION = os.environ.get("LEGAL_RAG_RUN_INTEGRATION_DB") == "1"

pytestmark = pytest.mark.integration


def _require_postgres() -> None:
    invalidate_engine_cache()
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")


def _ingest_chunk_basic_corpus() -> list:
    root = Path(__file__).resolve().parents[2]
    basic = root / "data/sample_corpus/basic"
    files = [
        basic / "intro.md",
        basic / "plain.txt",
        basic / "policies.md",
        basic / "sample.html",
    ]
    svc = default_ingestion_service()
    doc_ids: list = []
    for path in files:
        ing = ingest_file_persisted(path, svc)
        assert ing.error is None and ing.document and ing.document.id
        doc_ids.append(ing.document.id)

    cfg_ck = ChunkingConfig(strategy="fixed_size", chunk_size=400, chunk_overlap=40)
    for doc_id in doc_ids:
        c = chunk_document_persisted(doc_id, cfg_ck)
        assert c.error is None
    return doc_ids


@pytest.mark.skipif(not RUN_DB_INTEGRATION, reason="Set LEGAL_RAG_RUN_INTEGRATION_DB=1 to run")
def test_explicit_manifest_ignores_newer_unrelated_manifest() -> None:
    _require_postgres()
    root = Path(__file__).resolve().parents[2]
    golden_ret = root / "data/eval/retrieval_golden.jsonl"
    golden_ans = root / "data/eval/answer_golden.jsonl"

    _ingest_chunk_basic_corpus()

    ix_a = index_chunks_persisted(
        IndexingConfig(chunking_strategy="fixed_size", embedding_dimensions=16, batch_size=8)
    )
    assert ix_a.error is None and ix_a.manifest is not None
    mid_a = ix_a.manifest.id
    assert mid_a is not None

    settings = get_settings()
    cfg_a = RetrievalConfig(
        mode="hybrid",
        chunking_strategy="fixed_size",
        top_k=5,
        index_manifest_id=mid_a,
    )
    with session_scope(settings.database_url) as session:
        summary_a = RetrievalEvaluationRunner.from_session(session, cfg_a).run_file(
            golden_ret, top_k=5
        )

    assert summary_a.manifest.get("id") == str(mid_a)
    assert summary_a.execution_status == "PASSED"
    hit_rate_a = summary_a.hit_rate
    mrr_a = summary_a.mrr
    hits_a = tuple((it.question_id, it.hit, it.hit_rank) for it in summary_a.items)

    ix_b = index_chunks_persisted(
        IndexingConfig(
            chunking_strategy="fixed_size",
            embedding_dimensions=16,
            batch_size=8,
            force_reindex=True,
        )
    )
    assert ix_b.error is None and ix_b.manifest is not None
    mid_b = ix_b.manifest.id
    assert mid_b is not None
    assert mid_b != mid_a

    cfg_auto = RetrievalConfig(mode="hybrid", chunking_strategy="fixed_size", top_k=5)
    with session_scope(settings.database_url) as session:
        auto_mf = resolve_manifest_record(session, cfg_auto, for_dense=True)
    assert auto_mf.id == mid_b

    with session_scope(settings.database_url) as session:
        summary_pinned = RetrievalEvaluationRunner.from_session(session, cfg_a).run_file(
            golden_ret, top_k=5
        )

    assert summary_pinned.manifest.get("id") == str(mid_a)
    assert summary_pinned.hit_rate == hit_rate_a
    assert summary_pinned.mrr == mrr_a
    hits_pinned = tuple(
        (it.question_id, it.hit, it.hit_rank) for it in summary_pinned.items
    )
    assert hits_pinned == hits_a

    cfg_ans = RetrievalConfig(
        mode="sparse_only",
        chunking_strategy="fixed_size",
        top_k=5,
        index_manifest_id=mid_a,
    )
    with session_scope(settings.database_url) as session:
        ans_a = AnswerEvaluationRunner.from_session(session, cfg_ans).run_file(
            golden_ans, top_k=5
        )
    assert ans_a.manifest.get("id") == str(mid_a)
    pass_rate_a = ans_a.pass_rate

    with session_scope(settings.database_url) as session:
        ans_pinned = AnswerEvaluationRunner.from_session(session, cfg_ans).run_file(
            golden_ans, top_k=5
        )
    assert ans_pinned.manifest.get("id") == str(mid_a)
    assert ans_pinned.pass_rate == pass_rate_a


@pytest.mark.skipif(not RUN_DB_INTEGRATION, reason="Set LEGAL_RAG_RUN_INTEGRATION_DB=1 to run")
def test_unknown_explicit_manifest_fails_without_fallback() -> None:
    _require_postgres()
    _ingest_chunk_basic_corpus()
    ix = index_chunks_persisted(
        IndexingConfig(chunking_strategy="fixed_size", embedding_dimensions=16, batch_size=8)
    )
    assert ix.error is None and ix.manifest is not None

    unknown = uuid4()
    cfg = RetrievalConfig(
        mode="hybrid",
        chunking_strategy="fixed_size",
        top_k=5,
        index_manifest_id=unknown,
    )
    settings = get_settings()
    with session_scope(settings.database_url) as session:
        with pytest.raises(ManifestNotFoundError, match=str(unknown)):
            RetrievalEvaluationRunner.from_session(session, cfg)

    with session_scope(settings.database_url) as session:
        with pytest.raises(ManifestNotFoundError, match=str(unknown)):
            AnswerEvaluationRunner.from_session(
                session,
                RetrievalConfig(
                    mode="sparse_only",
                    chunking_strategy="fixed_size",
                    top_k=5,
                    index_manifest_id=unknown,
                ),
            )
