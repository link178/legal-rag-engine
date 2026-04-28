"""Unit tests for deterministic hash embedding provider (no downloads)."""

from __future__ import annotations

from app.indexing.dense.mock_provider import DeterministicHashEmbeddingProvider


def test_same_text_same_vector() -> None:
    p = DeterministicHashEmbeddingProvider(dimensions=16)
    a = p.embed_query("hello world")
    b = p.embed_query("hello world")
    assert a == b
    assert len(a) == 16


def test_different_texts_usually_differ() -> None:
    p = DeterministicHashEmbeddingProvider(dimensions=32)
    a = p.embed_query("alpha")
    b = p.embed_query("beta")
    assert a != b
    assert len(a) == 32


def test_embed_query_matches_embed_texts_first() -> None:
    p = DeterministicHashEmbeddingProvider(dimensions=16)
    text = "coherent"
    q = p.embed_query(text)
    batch = p.embed_texts([text])
    assert batch[0] == q


def test_vectors_normalized_to_unit_length() -> None:
    p = DeterministicHashEmbeddingProvider(dimensions=8)
    v = p.embed_query("norm check")
    s = sum(x * x for x in v) ** 0.5
    assert abs(s - 1.0) < 1e-6
