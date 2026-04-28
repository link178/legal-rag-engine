"""Unit tests for ``ChunkingConfig`` (no DB)."""

from __future__ import annotations

import pytest
from app.chunking.models import ChunkingConfig


def test_default_config_valid() -> None:
    c = ChunkingConfig()
    assert c.strategy == "fixed_size"
    assert c.chunk_size == 1200
    assert c.chunk_overlap == 200
    assert c.min_chunk_chars == 80
    assert c.preserve_headings is True
    assert len(c.config_hash()) == 64


def test_rejects_overlap_gte_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)


def test_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        ChunkingConfig(chunk_size=0)


def test_normalized_dict_sorted_for_hash_stability() -> None:
    c1 = ChunkingConfig(strategy="fixed_size", chunk_size=500, chunk_overlap=50)
    c2 = ChunkingConfig(strategy="fixed_size", chunk_overlap=50, chunk_size=500)
    assert c1.normalized_dict() == c2.normalized_dict()
    assert c1.config_hash() == c2.config_hash()
