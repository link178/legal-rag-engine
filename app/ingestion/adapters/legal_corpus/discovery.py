"""Recursive corpus file discovery with stable ordering and ignore rules."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.adapters.legal_corpus.models import LegalCorpusDocumentCandidate
from app.ingestion.parsers.metadata import infer_source_type

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}
)

_IGNORE_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
)


def _relative_has_ignored_parent(rel: Path) -> bool:
    """True if any directory component under the corpus root should be skipped."""
    parts = rel.parts[:-1] if len(rel.parts) > 1 else ()
    for part in parts:
        if part.startswith(".") or part in _IGNORE_DIR_NAMES:
            return True
    return False


def scan_corpus(corpus_path: Path) -> tuple[LegalCorpusDocumentCandidate, ...]:
    """Walk ``corpus_path`` recursively and list supported files in deterministic order.

    Skips symlinks, hidden files, ignored directory subtrees, and unsupported extensions.
    ``relative_path`` uses POSIX separators for stable sorting across platforms.
    """
    root = corpus_path.expanduser().resolve()
    if not root.is_dir():
        return ()

    candidates: list[LegalCorpusDocumentCandidate] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.is_symlink():
            continue
        if path.name.startswith("."):
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if _relative_has_ignored_parent(rel):
            continue
        ext = path.suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            continue
        rel_posix = rel.as_posix()
        candidates.append(
            LegalCorpusDocumentCandidate(
                path=path,
                relative_path=rel_posix,
                extension=ext,
                inferred_source_type=infer_source_type(path),
            )
        )

    candidates.sort(key=lambda c: c.relative_path)
    return tuple(candidates)
