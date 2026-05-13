"""Frontmatter parsing (minimal) and legal corpus metadata heuristics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.ingestion.adapters.legal_corpus.models import LegalCorpusDocumentCandidate

_CORPUS_ADAPTER = "legalize"


def parse_frontmatter(raw_text: str) -> tuple[dict[str, Any], list[str]]:
    """Parse a leading YAML-like ``---`` block into scalar key/value pairs only.

    Does not use PyYAML. Booleans and integers are recognized; everything else is ``str``.
    Unrecognized lines add warnings; malformed blocks do not raise.
    """
    warnings: list[str] = []
    if not raw_text.startswith("---"):
        return {}, warnings

    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, warnings

    end_idx: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        warnings.append("frontmatter: unclosed delimiter")
        return {}, warnings

    block_lines = lines[1:end_idx]
    result: dict[str, Any] = {}
    kv_pattern = re.compile(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)\s*$")

    for raw_line in block_lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            warnings.append("frontmatter: skipped comment line")
            continue
        m = kv_pattern.match(line)
        if not m:
            warnings.append(f"frontmatter: skipped non key-value line: {line[:60]!r}")
            continue
        key, value_str = m.group(1), m.group(2)
        value_str = value_str.strip()
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            value_str = value_str[1:-1]
        low = value_str.lower()
        if low in ("true", "false"):
            parsed: Any = low == "true"
        elif value_str.isdigit():
            parsed = int(value_str)
        else:
            parsed = value_str
        result[key] = parsed

    return result, warnings


def _infer_jurisdiction(parts_lower: tuple[str, ...]) -> str:
    for p in parts_lower:
        if "legalize-es" in p or p == "es":
            return "es"
    for p in parts_lower:
        if "legalize-eu" in p or p == "eu":
            return "eu"
    return "unknown"


def _infer_legal_document_type(filename_lower: str) -> str:
    if "royal-decree" in filename_lower or "real-decreto" in filename_lower:
        return "royal_decree"
    if "regulation" in filename_lower:
        return "regulation"
    if "directive" in filename_lower:
        return "directive"
    if "law" in filename_lower or "ley" in filename_lower:
        return "law"
    if "guide" in filename_lower or "guidance" in filename_lower:
        return "guidance"
    return "unknown"


def infer_corpus_metadata(
    corpus_path: Path,
    candidate: LegalCorpusDocumentCandidate,
    frontmatter: dict[str, Any],
) -> dict[str, Any]:
    """Build corpus-level metadata; ``frontmatter`` values override heuristics."""
    root = corpus_path.expanduser().resolve()
    try:
        rel = candidate.path.relative_to(root)
    except ValueError:
        rel = Path(candidate.relative_path)

    parts_lower = tuple(p.lower() for p in rel.parts)
    jurisdiction = _infer_jurisdiction(parts_lower)
    legal_type = _infer_legal_document_type(candidate.path.name.lower())

    if "language" in frontmatter:
        language: Any = frontmatter["language"]
        if not isinstance(language, str):
            language = str(language)
    elif jurisdiction == "es":
        language = "es"
    else:
        language = "unknown"

    meta: dict[str, Any] = {
        "corpus_adapter": _CORPUS_ADAPTER,
        "corpus_name": root.name,
        "corpus_relative_path": candidate.relative_path,
        "jurisdiction": jurisdiction,
        "source_family": _CORPUS_ADAPTER,
        "legal_document_type": legal_type,
        "language": language,
        "version": None,
        "effective_date": None,
        "canonical_id": None,
    }

    fm_doc_type = frontmatter.get("document_type")
    if fm_doc_type is not None:
        meta["legal_document_type"] = str(fm_doc_type)

    if "jurisdiction" in frontmatter:
        meta["jurisdiction"] = str(frontmatter["jurisdiction"])

    if "language" in frontmatter:
        meta["language"] = str(frontmatter["language"])

    if "version" in frontmatter:
        meta["version"] = frontmatter["version"]

    if "effective_date" in frontmatter:
        meta["effective_date"] = str(frontmatter["effective_date"])

    if "canonical_id" in frontmatter:
        meta["canonical_id"] = str(frontmatter["canonical_id"])

    # Optional informational title from frontmatter (loader still sets Document.title)
    if "title" in frontmatter:
        meta["title"] = str(frontmatter["title"])

    return meta


def load_frontmatter_for_candidate(
    candidate: LegalCorpusDocumentCandidate,
) -> tuple[dict[str, Any], list[str]]:
    """Read leading frontmatter from Markdown files; other types return empty."""
    if candidate.extension not in (".md", ".markdown"):
        return {}, []
    try:
        raw = candidate.path.read_text(encoding="utf-8")
    except OSError as e:
        return {}, [f"frontmatter: could not read file: {e}"]
    return parse_frontmatter(raw)
