"""Term frequencies from raw chunk text (initial sparse representation, no retrieval yet)."""

from __future__ import annotations

from app.indexing.sparse.tokenizer import tokenize


def extract_sparse_terms(text: str) -> dict[str, int]:
    """Return lowercase term -> count for indexing metadata (not Postgres FTS yet)."""
    counts: dict[str, int] = {}
    for tok in tokenize(text):
        counts[tok] = counts.get(tok, 0) + 1
    return counts
