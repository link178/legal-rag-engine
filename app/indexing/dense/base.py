"""Embedding provider protocol (dependency-free); real local models deferred to Phase 4B."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Compute dense vectors from text batches."""

    @property
    def name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding per input string (same order)."""

    def embed_query(self, text: str) -> list[float]:
        """Single-vector query embedding."""
