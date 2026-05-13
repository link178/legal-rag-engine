"""RetrievalConfig and result model validation."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest
from app.retrieval.models import (
    RetrievalConfig,
    RetrievalMetadataFilter,
    RetrievalResultSet,
    RetrievedChunk,
)


def test_retrieval_config_defaults() -> None:
    c = RetrievalConfig()
    assert c.mode == "hybrid"
    assert c.top_k == 5
    assert c.dense_top_k == 10
    assert c.sparse_top_k == 10
    assert c.rrf_k == 60


def test_retrieval_config_rejects_bad_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        RetrievalConfig(mode="bogus")


def test_retrieval_config_rejects_nonpositive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        RetrievalConfig(top_k=0)
    with pytest.raises(ValueError, match="rrf_k"):
        RetrievalConfig(rrf_k=0)


def test_retrieval_config_rejects_bad_embedding_dimensions() -> None:
    with pytest.raises(ValueError, match="embedding_dimensions"):
        RetrievalConfig(embedding_dimensions=0)


def test_retrieved_chunk_and_result_set() -> None:
    cid = uuid4()
    did = uuid4()
    ch = RetrievedChunk(
        chunk_id=cid,
        document_id=did,
        text="hello",
        source_path="a.md",
        title="T",
        heading="H",
        chunk_index=0,
        chunking_strategy="fixed_size",
        dense_score=0.5,
        rank_position=1,
        retrieval_sources=("dense",),
    )
    rs = RetrievalResultSet(
        query="q",
        mode="dense_only",
        results=[ch],
        index_manifest_id=uuid4(),
        manifest_hash="x" * 64,
    )
    assert rs.results[0].chunk_id == cid


def test_retrieval_config_metadata_filter_roundtrip() -> None:
    mf = RetrievalMetadataFilter(jurisdiction="eu")
    c = RetrievalConfig(metadata_filter=mf, mode="sparse_only")
    assert c.metadata_filter is mf
    c2 = replace(c, top_k=3)
    assert c2.metadata_filter is mf
    assert c2.top_k == 3

