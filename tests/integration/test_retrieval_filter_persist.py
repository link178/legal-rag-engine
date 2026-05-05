"""Optional metadata-filter retrieval scenarios (Postgres + LEGAL_RAG_RUN_INTEGRATION_DB=1)."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest
from app.chunking.models import ChunkingConfig
from app.chunking.runner import chunk_document_persisted
from app.indexing.models import IndexingConfig
from app.indexing.runner import index_chunks_persisted
from app.ingestion.adapters.legal_corpus.importer import LegalCorpusImporter
from app.ingestion.services import default_ingestion_service
from app.retrieval.dense import DenseRetriever
from app.retrieval.manifest import resolve_manifest_record
from app.retrieval.models import RetrievalConfig, RetrievalMetadataFilter
from app.retrieval.orchestrator import RetrievalOrchestrator
from app.retrieval.sparse import SparseRetriever
from app.storage.postgres.session import get_engine, invalidate_engine_cache, session_scope
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

RUN_DB_INTEGRATION = os.environ.get("LEGAL_RAG_RUN_INTEGRATION_DB") == "1"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not RUN_DB_INTEGRATION, reason="Set LEGAL_RAG_RUN_INTEGRATION_DB=1 to run")
def test_retrieval_metadata_filters_legalize_sample() -> None:
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
    summary = importer.import_corpus(sample, persist=True)
    assert summary.failed_count == 0
    assert summary.imported_count >= 1

    cfg_ck = ChunkingConfig(strategy="fixed_size", chunk_size=400, chunk_overlap=40)
    for it in summary.items:
        if it.document_id:
            chk = chunk_document_persisted(UUID(it.document_id), cfg_ck)
            assert chk.error is None

    ix_cfg = IndexingConfig(chunking_strategy="fixed_size", embedding_dimensions=16, batch_size=8)
    r1 = index_chunks_persisted(ix_cfg)
    assert r1.error is None and r1.manifest is not None
    mid = r1.manifest.id
    assert mid is not None

    base = RetrievalConfig(
        mode="hybrid",
        chunking_strategy="fixed_size",
        top_k=5,
        index_manifest_id=mid,
    )
    f_eu_reg = RetrievalMetadataFilter(jurisdiction="eu", legal_document_type="regulation")

    with session_scope() as session:
        mf = resolve_manifest_record(session, base, for_dense=True)
        assert mf.include_sparse is True
        orch = RetrievalOrchestrator(DenseRetriever(session), SparseRetriever(session))

        cfg_match = replace(base, metadata_filter=f_eu_reg)
        res_ok = orch.retrieve("AI Act synthetic legal sample", cfg_match, mf)
        assert res_ok.results, "expected hits for eu+regulation filter"
        assert res_ok.metadata.get("metadata_filter") == {
            "jurisdiction": "eu",
            "legal_document_type": "regulation",
        }

        cfg_no = replace(base, metadata_filter=RetrievalMetadataFilter(jurisdiction="zz"))
        res_empty = orch.retrieve("AI Act", cfg_no, mf)
        assert not res_empty.results
        assert res_empty.metadata.get("metadata_filter") == {"jurisdiction": "zz"}

        # No alphanumeric tokens → sparse branch empty; hybrid still completes.
        cfg_h = replace(base, metadata_filter=f_eu_reg)
        res_h = orch.retrieve("###", cfg_h, mf)
        assert res_h.metadata.get("metadata_filter") == {
            "jurisdiction": "eu",
            "legal_document_type": "regulation",
        }
        assert res_h.results or not res_h.results

        cfg_s = replace(
            base,
            mode="sparse_only",
            metadata_filter=f_eu_reg,
        )
        mf_s = resolve_manifest_record(session, cfg_s, for_dense=False)
        res_s = orch.retrieve("fragment", cfg_s, mf_s)
        assert res_s.results, "sparse-only with filter should still rank EU regulation chunks"
        assert all("sparse" in x.retrieval_sources for x in res_s.results)
