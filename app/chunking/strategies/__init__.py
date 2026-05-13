"""Chunking strategies."""

from __future__ import annotations

from app.chunking.strategies.fixed_size import STRATEGY_NAME as FIXED_SIZE
from app.chunking.strategies.fixed_size import FixedSizeChunkingStrategy
from app.chunking.strategies.structure_aware import STRATEGY_NAME as STRUCTURE_AWARE
from app.chunking.strategies.structure_aware import StructureAwareChunkingStrategy

__all__ = [
    "FIXED_SIZE",
    "STRUCTURE_AWARE",
    "FixedSizeChunkingStrategy",
    "StructureAwareChunkingStrategy",
]
