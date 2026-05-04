"""Grounded prompt assembly."""

from __future__ import annotations

from uuid import uuid4

from app.generation.models import GroundedContextBlock
from app.generation.prompts import build_grounded_prompt
from app.generation.providers.base import INSUFFICIENT_CONTEXT_SENTENCE


def _b(cid: int) -> GroundedContextBlock:
    return GroundedContextBlock(
        citation_id=cid,
        chunk_id=uuid4(),
        document_id=uuid4(),
        source_path="intro.md",
        title="Intro",
        heading="Section",
        rank=cid,
        score=0.5,
        text=f"Evidence {cid}.",
    )


def test_prompt_contains_question_instructions_and_blocks() -> None:
    prompt = build_grounded_prompt(
        " What is X? ",
        [_b(1), _b(2)],
        insufficient_sentence=INSUFFICIENT_CONTEXT_SENTENCE,
    )
    assert "What is X?" in prompt
    assert INSUFFICIENT_CONTEXT_SENTENCE in prompt
    assert "Do not invent facts" in prompt
    assert "[1]" in prompt and "source: intro.md" in prompt
    assert "title: Intro" in prompt and "heading: Section" in prompt
    assert "[2]" in prompt
    assert "Evidence 1." in prompt


def test_prompt_uses_em_dash_for_missing_metadata() -> None:
    b = GroundedContextBlock(
        citation_id=1,
        chunk_id=uuid4(),
        document_id=uuid4(),
        source_path=None,
        title=None,
        heading=None,
        rank=1,
        score=None,
        text="body",
    )
    p = build_grounded_prompt("q?", [b], insufficient_sentence=INSUFFICIENT_CONTEXT_SENTENCE)
    assert "source: —" in p


def test_empty_context_section() -> None:
    p = build_grounded_prompt("x?", (), insufficient_sentence=INSUFFICIENT_CONTEXT_SENTENCE)
    assert "(none)" in p
    assert "Context:" in p
