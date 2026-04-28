"""File metadata helpers (deterministic where possible)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def infer_source_type(path: Path) -> str:
    """Return canonical ``source_type`` for supported extensions."""
    ext = path.suffix.lower()
    if ext == ".txt":
        return "text"
    if ext in (".md", ".markdown"):
        return "markdown"
    return "unknown"


def extract_basic_file_metadata(path: Path) -> dict[str, Any]:
    """Stable file metadata (no wall-clock fields that break tests)."""
    stat = path.stat()
    return {
        "filename": path.name,
        "extension": path.suffix.lower().lstrip(".") or None,
        "file_size_bytes": stat.st_size,
    }
