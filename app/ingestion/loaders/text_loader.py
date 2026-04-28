"""Plain text loader (.txt)."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.errors import DocumentLoadError
from app.ingestion.loaders.base import LoadedDocument


class TextLoader:
    supported_extensions: tuple[str, ...] = (".txt",)

    def load(self, path: Path) -> LoadedDocument:
        try:
            raw_bytes = path.read_bytes()
        except OSError as e:
            raise DocumentLoadError(f"Failed to read file: {e}") from e
        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise DocumentLoadError(f"File is not valid UTF-8: {e}") from e
        stem = path.stem
        title = stem if stem else None
        return LoadedDocument(
            source_path=str(path.resolve()),
            source_type="text",
            raw_text=raw_text,
            title=title,
            metadata={},
        )
