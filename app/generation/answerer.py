"""Answer synthesis from retrieved context (placeholder)."""

from typing import Any


class Answerer:
    """Produce grounded answers; implementation deferred."""

    def answer(self, query: str, contexts: list[Any]) -> str:
        """Generate answer text (not implemented)."""
        raise NotImplementedError
