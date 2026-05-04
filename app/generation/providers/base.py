"""Generation provider protocol."""

from __future__ import annotations

from typing import Protocol

INSUFFICIENT_CONTEXT_SENTENCE = (
    "I do not have enough information in the provided context to answer this question."
)


class GenerationProvider(Protocol):
    """Single-call text generator (mock or future LLM adapter)."""

    name: str

    def generate(self, prompt: str) -> str:
        """Return model output for the full prompt string."""
