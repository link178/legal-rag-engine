"""Simple alphanumeric tokenization for sparse term counts (no external NLP deps)."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[0-9a-z]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase tokens: sequences of letters/digits split on non-alphanumeric."""
    if not text:
        return []
    lowered = text.lower()
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(lowered)]
