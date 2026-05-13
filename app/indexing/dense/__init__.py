"""Dense (vector) indexing: provider protocol and deterministic test provider."""

from __future__ import annotations

from app.indexing.dense.base import EmbeddingProvider
from app.indexing.dense.mock_provider import DeterministicHashEmbeddingProvider

__all__ = ["EmbeddingProvider", "DeterministicHashEmbeddingProvider"]
