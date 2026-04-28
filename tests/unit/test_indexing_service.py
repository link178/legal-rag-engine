"""IndexingService pure unit tests (no DB)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from app.domain.models import Chunk
from app.indexing.models import IndexingConfig
from app.indexing.service import default_indexing_service


def _chunk(text: str, strategy: str = "fixed_size") -> Chunk:
    doc_id = uuid4()
    return Chunk(
        id=uuid4(),
        document_id=doc_id,
        chunk_index=0,
        text=text,
        chunking_strategy=strategy,
        char_count=len(text),
    )


def test_index_dense_and_sparse() -> None:
    svc = default_indexing_service(dimensions=16)
    cfg = IndexingConfig(include_dense=True, include_sparse=True)
    ch = [_chunk("hello legal world")]
    results, dense, sparse = svc.index_chunks(ch, cfg)
    assert len(results) == 1
    assert results[0].dense_indexed and results[0].sparse_indexed
    assert dense is not None and len(dense[0]) == 16
    assert sparse is not None and sparse[0]["legal"] == 1


def test_index_sparse_only() -> None:
    svc = default_indexing_service(dimensions=8)
    cfg = IndexingConfig(include_dense=False, include_sparse=True)
    results, dense, sparse = svc.index_chunks([_chunk("a b a")], cfg)
    assert dense is None
    assert sparse is not None
    assert results[0].sparse_indexed and not results[0].dense_indexed


def test_index_dense_only() -> None:
    svc = default_indexing_service(dimensions=4)
    cfg = IndexingConfig(include_dense=True, include_sparse=False)
    results, dense, sparse = svc.index_chunks([_chunk("only dense")], cfg)
    assert sparse is None
    assert dense is not None
    assert results[0].dense_indexed and not results[0].sparse_indexed


def test_requires_chunk_id() -> None:
    svc = default_indexing_service()
    bad = Chunk(
        document_id=uuid4(),
        chunk_index=0,
        text="missing id",
        chunking_strategy="fixed_size",
        char_count=9,
        id=None,
    )
    with pytest.raises(ValueError, match="chunk id"):
        svc.index_chunks([bad], IndexingConfig())
