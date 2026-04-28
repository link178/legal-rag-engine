"""Coordinates retrieval pipeline stages (placeholder)."""

from typing import Any


class RetrievalOrchestrator:
    """Dense → sparse → fusion → optional rerank wiring (future)."""

    def run(self, query: str, **kwargs: Any) -> list[Any]:
        """Execute retrieval for a query (not implemented)."""
        raise NotImplementedError
