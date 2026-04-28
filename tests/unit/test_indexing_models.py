"""IndexingConfig and manifest hash helpers."""

from __future__ import annotations

from uuid import uuid4

import pytest
from app.indexing.models import (
    IndexingConfig,
    chunk_set_hash_from_fingerprints,
    fingerprint_chunk_row,
    manifest_hash,
)


def test_indexing_config_defaults() -> None:
    c = IndexingConfig()
    assert c.embedding_provider == "deterministic_hash"
    assert c.embedding_dimensions == 16
    assert c.batch_size == 32
    assert c.include_sparse is True
    assert c.include_dense is True


def test_rejects_batch_size_zero() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        IndexingConfig(batch_size=0)


def test_rejects_both_branches_off() -> None:
    with pytest.raises(ValueError, match="include_sparse"):
        IndexingConfig(include_sparse=False, include_dense=False)


def test_rejects_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="embedding_dimensions"):
        IndexingConfig(embedding_dimensions=0)


def test_config_hash_stable() -> None:
    a = IndexingConfig(chunking_strategy="fixed_size")
    b = IndexingConfig(chunking_strategy="fixed_size")
    assert a.config_hash() == b.config_hash()


def test_chunk_set_hash_order_independent() -> None:
    u1 = uuid4()
    u2 = uuid4()
    fp1 = fingerprint_chunk_row(u1, chunking_strategy="fixed_size", checksum="aaa")
    fp2 = fingerprint_chunk_row(u2, chunking_strategy="fixed_size", checksum="bbb")
    h_ab = chunk_set_hash_from_fingerprints([fp1, fp2])
    h_ba = chunk_set_hash_from_fingerprints([fp2, fp1])
    assert h_ab == h_ba


def test_manifest_hash_combines() -> None:
    m = manifest_hash("cfg", "chunks")
    assert len(m) == 64
