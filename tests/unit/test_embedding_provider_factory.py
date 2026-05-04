"""Unit tests for ``build_embedding_provider`` (no optional deps required)."""

from __future__ import annotations

import sys

import pytest
from app.indexing.dense.factory import build_embedding_provider
from app.indexing.dense.mock_provider import DeterministicHashEmbeddingProvider
from app.indexing.models import IndexingConfig


def test_factory_deterministic() -> None:
    cfg = IndexingConfig(embedding_dimensions=8)
    p = build_embedding_provider(cfg)
    assert isinstance(p, DeterministicHashEmbeddingProvider)
    assert p.dimensions == 8
    v = p.embed_query("x")
    assert len(v) == 8


def test_factory_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding_provider"):
        build_embedding_provider(
            IndexingConfig(embedding_provider="openai", embedding_dimensions=16)
        )


def test_factory_local_requires_model() -> None:
    with pytest.raises(ValueError, match="embedding_model is required"):
        build_embedding_provider(
            IndexingConfig(
                embedding_provider="local_sentence_transformers",
                embedding_dimensions=384,
                embedding_model=None,
            )
        )


def test_factory_deterministic_does_not_import_sentence_transformers() -> None:
    sys.modules.pop("sentence_transformers", None)
    build_embedding_provider(IndexingConfig(embedding_dimensions=4))
    assert "sentence_transformers" not in sys.modules
