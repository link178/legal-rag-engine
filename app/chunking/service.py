"""Coordinates chunking strategies and exposes ``chunk_document``."""

from __future__ import annotations

from collections.abc import Mapping

from app.chunking.errors import UnsupportedChunkingStrategyError
from app.chunking.models import ChunkingConfig
from app.chunking.strategies.base import ChunkingStrategy
from app.chunking.strategies.fixed_size import FixedSizeChunkingStrategy
from app.chunking.strategies.structure_aware import StructureAwareChunkingStrategy
from app.domain.models import Chunk, Document


class ChunkingService:
    """Selects a strategy by name and splits a document into ``Chunk`` rows."""

    def __init__(self, strategies: Mapping[str, ChunkingStrategy]) -> None:
        self._strategies = dict(strategies)

    def chunk_document(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        name = config.strategy.strip()
        strategy = self._strategies.get(name)
        if strategy is None:
            supported = sorted(self._strategies)
            raise UnsupportedChunkingStrategyError(
                f"Unsupported chunking strategy {name!r}; supported: {supported}"
            )
        return strategy.split(document, config)


def default_chunking_service() -> ChunkingService:
    """Default service with fixed-size and structure-aware strategies."""
    fixed = FixedSizeChunkingStrategy()
    struct = StructureAwareChunkingStrategy()
    return ChunkingService({fixed.name: fixed, struct.name: struct})
