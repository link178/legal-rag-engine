"""Pure ingestion: path -> domain ``Document`` (no database)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.domain.models import Document
from app.ingestion.checksums import compute_file_checksum, compute_text_checksum
from app.ingestion.errors import (
    DocumentLoadError,
    DocumentNotFoundError,
    UnsupportedDocumentTypeError,
)
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.normalizers.text_normalizer import normalize_text
from app.ingestion.parsers.metadata import extract_basic_file_metadata


class IngestionService:
    """Select loader, normalize text, compute checksum, build ``Document``."""

    def __init__(self, loaders: Sequence[DocumentLoader]) -> None:
        self._loaders = list(loaders)
        self._ext_to_loader: dict[str, DocumentLoader] = {}
        for loader in self._loaders:
            for ext in loader.supported_extensions:
                e = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                self._ext_to_loader[e] = loader

    def ingest_file(self, path: Path) -> Document:
        p = path.expanduser()
        if not p.exists():
            raise DocumentNotFoundError(f"Path does not exist: {path}")
        if not p.is_file():
            raise DocumentNotFoundError(f"Path is not a file: {path}")

        ext = p.suffix.lower()
        loader = self._ext_to_loader.get(ext)
        if loader is None:
            raise UnsupportedDocumentTypeError(
                f"No loader for extension {ext!r}; supported: {sorted(self._ext_to_loader)}"
            )

        try:
            loaded = loader.load(p)
        except DocumentLoadError:
            raise

        normalized = normalize_text(loaded.raw_text)
        checksum = compute_text_checksum(normalized)
        file_meta = extract_basic_file_metadata(p)
        file_checksum = compute_file_checksum(p)

        metadata: dict = {
            **file_meta,
            **loaded.metadata,
            "file_checksum_sha256": file_checksum,
        }

        return Document(
            source_path=loaded.source_path,
            source_type=loaded.source_type,
            checksum=checksum,
            title=loaded.title,
            raw_text=loaded.raw_text,
            normalized_text=normalized,
            metadata=metadata,
        )


def default_ingestion_service() -> IngestionService:
    """Service with .txt, .md, and .markdown loaders."""
    from app.ingestion.loaders.markdown_loader import MarkdownLoader
    from app.ingestion.loaders.text_loader import TextLoader

    return IngestionService([TextLoader(), MarkdownLoader()])
