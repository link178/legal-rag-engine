"""Chunking-related exceptions (no SQLAlchemy)."""

from __future__ import annotations


class ChunkingError(Exception):
    """Base error for chunking pipeline failures."""


class EmptyDocumentError(ChunkingError):
    """Raised when a document has no usable text for chunking."""


class UnsupportedChunkingStrategyError(ChunkingError):
    """Raised when the requested strategy name is not registered."""


class DocumentIdRequiredError(ChunkingError):
    """Raised when producing persisted-style chunks requires ``Document.id`` and it is missing."""


class ChunkingConfigConflictError(ChunkingError):
    """Persisted chunks exist for the same document/strategy with a different config hash."""
