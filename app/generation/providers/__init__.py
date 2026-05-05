"""Generation providers (protocol + mock baseline)."""

from __future__ import annotations

from app.generation.providers.base import INSUFFICIENT_CONTEXT_SENTENCE, GenerationProvider
from app.generation.providers.mock import MockGenerationProvider

__all__ = [
    "INSUFFICIENT_CONTEXT_SENTENCE",
    "GenerationProvider",
    "MockGenerationProvider",
]
