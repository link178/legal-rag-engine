"""Optional end-to-end generation smoke (Postgres + LEGAL_RAG_RUN_INTEGRATION_DB=1)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from app.chunking.models import ChunkingConfig
from app.chunking.runner import chunk_document_persisted
from app.core.config import get_settings
from app.generation.answerer import GroundedAnswerer
from app.generation.cli import answer_to_dict
from app.generation.context import ContextBuilder
from app.generation.providers.mock import MockGenerationProvider
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
def test_generation_smoke_intro_hybrid_serializable() -> None:
    invalidate_engine_cache()
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")

    with session_scope() as session:
        n_before = session.scalar(
            select(func.count()).select_from(ProcessingRunRecord).where(
                ProcessingRunRecord.run_type == "generation"
            )
        )
    assert n_before == 0

    root = Path(__file__).resolve().parents[2]
    intro = root / "data/sample_corpus/basic/intro.md"
    svc = default_ingestion_service()
    ing = ingest_file_persisted(intro, svc)
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

    settings = get_settings()
    cfg = RetrievalConfig(
        mode="hybrid",
        chunking_strategy="fixed_size",
        top_k=5,
        index_manifest_id=mid,
        embedding_dimensions=settings.embedding_dimensions,
        embedding_provider=settings.embedding_provider,
        embedding_model=(settings.embedding_model or "").strip() or None,
    )

    with session_scope(settings.database_url) as session:
        ga = GroundedAnswerer.from_session(
            session,
            cfg,
            context_builder=ContextBuilder(),
            provider=MockGenerationProvider(),
        )
        out = ga.answer("What does the basic intro file describe?")

    assert out.mode in ("grounded", "partial", "insufficient_context")
    if out.mode == "grounded":
        assert len(out.citations) >= 1 and out.used_citation_ids

    json.dumps(answer_to_dict(out))

    with session_scope() as session:
        n_after = session.scalar(
            select(func.count()).select_from(ProcessingRunRecord).where(
                ProcessingRunRecord.run_type == "generation"
            )
        )
    assert n_after == 0
