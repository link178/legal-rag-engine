"""Conservative text normalization (preserves structure; no aggressive cleanup)."""

from __future__ import annotations


def normalize_text(text: str) -> str:
    """Normalize line endings, trim trailing spaces per line, collapse excessive blank lines.

    - CRLF / CR -> LF
    - Strip trailing spaces/tabs on each line (not leading indentation)
    - Replace runs of 3+ newlines with exactly 2 newlines
    - Final outer strip
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    stripped_lines = [line.rstrip(" \t") for line in lines]
    joined = "\n".join(stripped_lines)

    while "\n\n\n" in joined:
        joined = joined.replace("\n\n\n", "\n\n")

    return joined.strip()
