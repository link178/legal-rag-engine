"""Minimal Markdown loader (.md, .markdown) — first # heading only."""

from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.errors import DocumentLoadError
from app.ingestion.loaders.base import LoadedDocument

_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


class MarkdownLoader:
    supported_extensions: tuple[str, ...] = (".md", ".markdown")

    def load(self, path: Path) -> LoadedDocument:
        try:
            raw_bytes = path.read_bytes()
        except OSError as e:
            raise DocumentLoadError(f"Failed to read file: {e}") from e
        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise DocumentLoadError(f"File is not valid UTF-8: {e}") from e

        title: str | None = None
        m = _HEADING_RE.search(raw_text)
        if m:
            title = m.group(1).strip()
        if not title:
            stem = path.stem
            title = stem if stem else None

        return LoadedDocument(
            source_path=str(path.resolve()),
            source_type="markdown",
            raw_text=raw_text,
            title=title,
            metadata={},
        )
