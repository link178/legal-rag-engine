"""Deterministic pseudo-embeddings for tests and dev scaffolding — not semantic quality."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence


class DeterministicHashEmbeddingProvider:
    """
    Maps UTF-8 text to a fixed-size float vector via repeated SHA-256 blocks.

    **Not for production retrieval**: vectors do not represent real semantic similarity.
    Use only for Phase 4A pipelines, CI, and local smoke indexing without downloads.
    """

    def __init__(self, dimensions: int = 16, *, name: str = "deterministic_hash") -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be > 0")
        self._name = name
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _vec_for_text(self, text: str) -> list[float]:
        data = text.encode("utf-8")
        floats: list[float] = []
        counter = 0
        while len(floats) < self._dimensions:
            h = hashlib.sha256(data + counter.to_bytes(4, "big")).digest()
            # Two bytes -> one float in (-1, 1) scaled
            for i in range(0, len(h) - 1, 2):
                x = ((h[i] << 8) | h[i + 1]) / 65535.0 * 2.0 - 1.0
                floats.append(x)
                if len(floats) >= self._dimensions:
                    break
            counter += 1
        # L2-normalize for stable geometry in downstream distance code (future)
        norm = math.sqrt(sum(x * x for x in floats)) or 1.0
        return [x / norm for x in floats]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec_for_text(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        # Same routing as embed_texts([text])[0] for coherence
        return self._vec_for_text(text)
