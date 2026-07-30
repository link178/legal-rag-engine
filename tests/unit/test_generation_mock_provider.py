"""MockGenerationProvider determinism and evidence-aware refusal."""

from __future__ import annotations

from app.generation.providers.base import INSUFFICIENT_CONTEXT_SENTENCE
from app.generation.providers.mock import MockGenerationProvider


def test_deterministic_two_runs() -> None:
    p = MockGenerationProvider()
    s = (
        "Question: evidence summary about cats\n"
        "Context:\n"
        "[1] source: a | title: — | heading: —\n"
        "evidence summary about cats\n"
        "[2] source: b | title: — | heading: —\n"
        "more evidence summary\n"
    )
    a1 = p.generate(s)
    a2 = p.generate(s)
    assert a1 == a2
    assert "[1]" in a1 and "[2]" in a1


def test_no_blocks_returns_sentinel() -> None:
    p = MockGenerationProvider()
    assert p.generate("question only\nContext:\n(none)\n") == INSUFFICIENT_CONTEXT_SENTENCE


def test_uses_first_block_ids() -> None:
    p = MockGenerationProvider()
    s = (
        "Question: widget calibration procedure\n"
        "Context:\n"
        "[3] source: x | title: — | heading: —\n"
        "widget calibration procedure details\n"
    )
    out = p.generate(s)
    assert "[3]" in out


def test_irrelevant_context_returns_sentinel() -> None:
    p = MockGenerationProvider()
    s = (
        "Question: What is the capital of Atlantis according to the corpus?\n"
        "Context:\n"
        "[1] source: data/sample_corpus/basic/plain.txt | title: — | heading: —\n"
        "Hello from the sample corpus.\n"
    )
    assert p.generate(s) == INSUFFICIENT_CONTEXT_SENTENCE


def test_prompt_without_question_line_still_cites_blocks() -> None:
    """Unit fixtures that omit Question: keep prior cite-any-header behaviour."""
    p = MockGenerationProvider()
    s = "[3] source: x | title: — | heading: —\nbody\n"
    out = p.generate(s)
    assert "[3]" in out
