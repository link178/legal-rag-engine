"""Citation id extraction and mechanical citation verification."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence

from app.generation.models import CitationVerificationResult, GroundedContextBlock

_DIGIT_REF = re.compile(r"\[(\d+)\]")


def _iter_cited_ids(text: str) -> Iterator[int]:
    """Yield every bracketed numeric citation in appearance order (may repeat)."""
    for m in _DIGIT_REF.finditer(text):
        yield int(m.group(1))


def extract_cited_ids(text: str) -> tuple[int, ...]:
    """
    Return citation ids in order of first appearance (deduped).

    Matches only bracketed integers ``[1]``, ``[42]`` — not ``[abc]``.
    """
    seen: set[int] = set()
    out: list[int] = []
    for n in _iter_cited_ids(text):
        if n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


def verify_citations(
    answer_text: str,
    context_blocks: Sequence[GroundedContextBlock],
) -> CitationVerificationResult:
    """
    Compare citation ids in ``answer_text`` to ``context_blocks``. Pure; no DB/LLM.
    """
    used_raw = list(_iter_cited_ids(answer_text))
    seen_seen: set[int] = set()
    used_unique_list: list[int] = []
    for n in used_raw:
        if n not in seen_seen:
            seen_seen.add(n)
            used_unique_list.append(n)
    used_unique = tuple(used_unique_list)

    avail_unique = sorted({b.citation_id for b in context_blocks})
    available_tuple = tuple(avail_unique)
    avail_set = set(avail_unique)

    valid_list = [cid for cid in used_unique_list if cid in avail_set]
    invalid_list = [cid for cid in used_unique_list if cid not in avail_set]

    used_set = set(used_unique_list)
    unused_list = [a for a in avail_unique if a not in used_set]

    counts: dict[int, int] = {}
    duplicate_order: list[int] = []
    for cid in used_raw:
        counts[cid] = counts.get(cid, 0) + 1
        if counts[cid] == 2:
            duplicate_order.append(cid)
    duplicates = tuple(duplicate_order)

    if used_unique:
        rate = float(len(valid_list)) / float(len(used_unique))
    else:
        rate = 0.0

    has_citations = len(used_unique) > 0
    has_valid = len(valid_list) > 0
    has_invalid = len(invalid_list) > 0

    return CitationVerificationResult(
        used_citation_ids=used_unique,
        available_citation_ids=available_tuple,
        valid_citation_ids=tuple(valid_list),
        invalid_citation_ids=tuple(invalid_list),
        unused_citation_ids=tuple(unused_list),
        duplicate_citation_ids=duplicates,
        citation_validity_rate=rate,
        has_citations=has_citations,
        has_valid_citations=has_valid,
        has_invalid_citations=has_invalid,
    )
