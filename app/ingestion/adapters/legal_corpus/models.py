"""Datatypes for legal corpus discovery and import summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class LegalCorpusDocumentCandidate:
    """One discovered file under a corpus root."""

    path: Path
    relative_path: str  # POSIX path relative to corpus root
    extension: str  # lowercase suffix including dot, e.g. ".md"
    inferred_source_type: str


LegalCorpusImportStatus = Literal["loaded", "imported", "reused", "failed"]


@dataclass(frozen=True)
class LegalCorpusImportItem:
    """Per-file outcome from an import pass."""

    source_path: str
    relative_path: str
    status: LegalCorpusImportStatus
    document_id: str | None
    title: str | None
    metadata: dict[str, Any]
    error: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LegalCorpusImportSummary:
    """Aggregate result of importing a corpus directory."""

    corpus_path: str
    corpus_name: str
    discovered_count: int
    imported_count: int
    reused_count: int
    failed_count: int
    items: tuple[LegalCorpusImportItem, ...]
    corpus_run_id: str | None = None
