"""Loader protocol and internal loaded-document DTO."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class LoadedDocument:
    """Short-lived result of reading a file before normalization."""

    source_path: str
    source_type: str
    raw_text: str
    title: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader(Protocol):
    """Loads raw text from a path for a set of extensions."""

    supported_extensions: tuple[str, ...]

    def load(self, path: Path) -> LoadedDocument: ...
