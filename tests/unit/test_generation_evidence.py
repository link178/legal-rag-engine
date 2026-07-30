"""Unit tests for deterministic evidence sufficiency."""

from __future__ import annotations

from uuid import uuid4

from app.generation.evidence import (
    content_terms,
    evidence_is_sufficient,
    matched_content_terms,
)
from app.generation.models import GroundedContextBlock


def _block(
    text: str,
    *,
    source_path: str | None = "p.md",
    citation_id: int = 1,
) -> GroundedContextBlock:
    return GroundedContextBlock(
        citation_id=citation_id,
        chunk_id=uuid4(),
        document_id=uuid4(),
        source_path=source_path,
        title=None,
        heading=None,
        rank=citation_id,
        score=0.5,
        text=text,
    )


def test_content_terms_drop_stopwords() -> None:
    assert "the" not in content_terms("What is the capital of Atlantis")
    assert "capital" in content_terms("What is the capital of Atlantis")
    assert "atlantis" in content_terms("What is the capital of Atlantis")


def test_sufficient_when_two_content_terms_match() -> None:
    blocks = [_block("Sample intro Markdown", source_path="basic/intro.md")]
    assert evidence_is_sufficient("What does the basic intro file describe?", blocks)


def test_insufficient_when_only_incidental_overlap() -> None:
    blocks = [
        _block(
            "Hello from the sample corpus.",
            source_path="data/sample_corpus/basic/plain.txt",
        )
    ]
    q = "What is the capital of Atlantis according to the corpus?"
    assert matched_content_terms(q, blocks) == ("corpus",)
    assert evidence_is_sufficient(q, blocks) is False


def test_insufficient_with_empty_blocks() -> None:
    assert evidence_is_sufficient("plain hello message", []) is False


def test_single_content_term_needs_one_match() -> None:
    blocks = [_block("hello world")]
    assert evidence_is_sufficient("hello", blocks) is True
    assert evidence_is_sufficient("goodbye", blocks) is False
