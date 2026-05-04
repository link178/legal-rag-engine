"""Citation id extraction (Phase 7 will add claim-level verification)."""

from __future__ import annotations

import re
from typing import Any

_DIGIT_REF = re.compile(r"\[(\d+)\]")


def extract_cited_ids(text: str) -> tuple[int, ...]:
    """
    Return citation ids in order of first appearance (deduped).

    Matches only bracketed integers ``[1]``, ``[42]`` — not ``[abc]``.
    """
    seen: set[int] = set()
    out: list[int] = []
    for m in _DIGIT_REF.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


def verify_citations(answer: str, contexts: list[Any]) -> list[Any]:
    """Align answer spans with source chunks (not implemented — Phase 7)."""
    raise NotImplementedError
