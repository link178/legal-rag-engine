"""Deterministic generator for tests and zero-cost operator demos."""

from __future__ import annotations

import re
from uuid import uuid4

from app.generation.evidence import evidence_is_sufficient
from app.generation.models import GroundedContextBlock
from app.generation.providers.base import INSUFFICIENT_CONTEXT_SENTENCE

_BLOCK_HEADER = re.compile(
    r"^\[(\d+)\]\s+source:\s*(.*?)\s*\|\s*title:\s*(.*?)\s*\|\s*heading:\s*(.*?)\s*$",
    re.MULTILINE,
)
_QUESTION_LINE = re.compile(r"^Question:\s*(.+)\s*$", re.MULTILINE)


def _parse_prompt_blocks(prompt: str) -> list[GroundedContextBlock]:
    """Rebuild minimal context blocks from a grounded prompt for evidence checks."""
    headers = list(_BLOCK_HEADER.finditer(prompt))
    if not headers:
        return []
    blocks: list[GroundedContextBlock] = []
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(prompt)
        body = prompt[start:end].strip()
        src_raw = m.group(2).strip()
        ttl_raw = m.group(3).strip()
        hdg_raw = m.group(4).strip()
        source_path = None if src_raw in {"", "\u2014", "—"} else src_raw
        title = None if ttl_raw in {"", "\u2014", "—"} else ttl_raw
        heading = None if hdg_raw in {"", "\u2014", "—"} else hdg_raw
        blocks.append(
            GroundedContextBlock(
                citation_id=int(m.group(1)),
                chunk_id=uuid4(),
                document_id=uuid4(),
                source_path=source_path,
                title=title,
                heading=heading,
                rank=i + 1,
                score=None,
                text=body,
            )
        )
    return blocks


class MockGenerationProvider:
    """Parse prompt blocks and emit a fixed answer only when evidence supports it."""

    name = "mock"

    def generate(self, prompt: str) -> str:
        q_match = _QUESTION_LINE.search(prompt)
        question = q_match.group(1).strip() if q_match else ""
        blocks = _parse_prompt_blocks(prompt)
        if not blocks:
            return INSUFFICIENT_CONTEXT_SENTENCE
        if question and not evidence_is_sufficient(question, blocks):
            return INSUFFICIENT_CONTEXT_SENTENCE
        # Prompts without a Question: line (unit fixtures) still cite present blocks.
        ids = [b.citation_id for b in blocks]
        take = ids[:3]
        refs = "".join(f"[{i}]" for i in take)
        return (
            "Based on the provided context, the answer is supported by the "
            f"retrieved evidence. {refs}"
        )
