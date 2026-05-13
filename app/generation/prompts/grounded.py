"""Build a deterministic grounded QA prompt string."""

from __future__ import annotations

from collections.abc import Sequence

from app.generation.models import GroundedContextBlock


def build_grounded_prompt(
    question: str,
    context_blocks: Sequence[GroundedContextBlock],
    *,
    insufficient_sentence: str,
) -> str:
    """Assemble instructions + question + numbered context blocks for the generator."""
    q = question.strip()
    header_lines = (
        "You answer only using the provided context.",
        "If the context does not support an answer, reply exactly:",
        insufficient_sentence.strip(),
        "Cite supporting evidence with bracketed ids like [1] or [1][2].",
        "Do not cite sources that are not listed. Do not invent facts.",
        "",
        f"Question: {q}",
        "",
        "Context:",
    )
    prefix = "\n".join(header_lines)

    if not context_blocks:
        return prefix + "\n(none)\n"

    body_parts: list[str] = []
    for b in context_blocks:
        src = _field(b.source_path)
        ttl = _field(b.title)
        hdg = _field(b.heading)
        block_header = (
            f"[{b.citation_id}] source: {src} | title: {ttl} | heading: {hdg}"
        )
        body_parts.extend([block_header, b.text.strip(), ""])
    body = "\n".join(body_parts).rstrip() + "\n"
    return prefix + "\n" + body


def _field(val: str | None) -> str:
    """Use em dash placeholder for absent optional fields (explicit, parse-friendly)."""
    if val is None or not val.strip():
        return "\u2014"
    return val.strip()
