"""MockGenerationProvider determinism."""

from __future__ import annotations

from app.generation.providers.base import INSUFFICIENT_CONTEXT_SENTENCE
from app.generation.providers.mock import MockGenerationProvider


def test_deterministic_two_runs() -> None:
    p = MockGenerationProvider()
    s = (
        "header\n"
        "[1] source: a | title: — | heading: —\n"
        "text\n"
        "[2] source: b | title: — | heading: —\n"
        "more\n"
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
    s = "[3] source: x | title: — | heading: —\nbody\n"
    out = p.generate(s)
    assert "[3]" in out
