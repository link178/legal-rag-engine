"""Chunking strategy protocol."""

from __future__ import annotations

from typing import Protocol

from app.chunking.models import ChunkingConfig
from app.domain.models import Chunk, Document


class ChunkingStrategy(Protocol):
    """Splits a document into domain ``Chunk`` instances."""

    name: str

    def split(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        """Return ordered chunks for ``document`` using ``config``."""
        ...
