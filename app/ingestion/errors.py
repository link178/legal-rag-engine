"""Controlled errors for the ingestion pipeline (no SQLAlchemy imports)."""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for ingestion failures."""


class DocumentNotFoundError(IngestionError):
    """Path does not exist or is not a file."""


class UnsupportedDocumentTypeError(IngestionError):
    """File extension is not supported by any registered loader."""


class DocumentLoadError(IngestionError):
    """Failed to read or decode document content."""
