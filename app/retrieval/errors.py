"""Retrieval pipeline errors (no SQLAlchemy imports)."""


class RetrievalError(Exception):
    """Base class for retrieval failures."""


class EmptyQueryError(RetrievalError):
    """Raised when the query string is empty or whitespace-only."""


class ManifestNotFoundError(RetrievalError):
    """No index manifest matches the retrieval filters or explicit id."""


class RetrieverNotConfiguredError(RetrievalError):
    """Required retriever for the chosen mode is missing."""


class EmbeddingDimensionMismatchError(RetrievalError):
    """Query embedding dimension does not match the manifest / stored vectors."""
