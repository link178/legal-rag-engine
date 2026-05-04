"""Grounded generation dataclass validation."""

from __future__ import annotations

from uuid import uuid4

import pytest
from app.generation.models import (
    GroundedAnswer,
    GroundedContextBlock,
    citation_from_block,
)


def _block(**kwargs) -> GroundedContextBlock:
    defaults = dict(
        citation_id=1,
        chunk_id=uuid4(),
        document_id=uuid4(),
        source_path="x.md",
        title="T",
        heading="H",
        rank=1,
        score=0.5,
        text="body",
    )
    defaults.update(kwargs)
    return GroundedContextBlock(**defaults)


def test_grounded_answer_valid() -> None:
    blk = _block()
    cit = citation_from_block(blk)
    ga = GroundedAnswer(
        question="q?",
        answer="a [1]",
        mode="grounded",
        citations=(cit,),
        used_citation_ids=(1,),
        retrieval_mode="hybrid",
        insufficient_context=False,
    )
    assert ga.mode == "grounded"
    assert ga.insufficient_context is False


def test_insufficient_context_answer() -> None:
    ga = GroundedAnswer(
        question="q?",
        answer="I do not have enough information in the provided context to answer this question.",
        mode="insufficient_context",
        citations=(),
        used_citation_ids=(),
        retrieval_mode="dense_only",
        insufficient_context=True,
    )
    assert ga.insufficient_context is True


def test_partial_with_citations_but_no_used_ids() -> None:
    blk = _block()
    cit = citation_from_block(blk)
    GroundedAnswer(
        question="q?",
        answer="unsupported claim",
        mode="partial",
        citations=(cit,),
        used_citation_ids=(),
        retrieval_mode="hybrid",
        insufficient_context=False,
    )


def test_empty_question_fails() -> None:
    blk = _block()
    cit = citation_from_block(blk)
    with pytest.raises(ValueError, match="question"):
        GroundedAnswer(
            question="  ",
            answer="a",
            mode="partial",
            citations=(cit,),
            used_citation_ids=(),
            retrieval_mode=None,
            insufficient_context=False,
        )


def test_empty_answer_fails() -> None:
    blk = _block()
    cit = citation_from_block(blk)
    with pytest.raises(ValueError, match="answer"):
        GroundedAnswer(
            question="q",
            answer=" \n",
            mode="partial",
            citations=(cit,),
            used_citation_ids=(),
            retrieval_mode=None,
            insufficient_context=False,
        )


def test_grounded_requires_matching_used_id() -> None:
    blk = _block()
    cit = citation_from_block(blk)
    with pytest.raises(ValueError, match="used_citation_id"):
        GroundedAnswer(
            question="q",
            answer="x [2]",
            mode="grounded",
            citations=(cit,),
            used_citation_ids=(2,),
            retrieval_mode=None,
            insufficient_context=False,
        )


def test_insufficient_must_have_empty_citations() -> None:
    blk = _block()
    cit = citation_from_block(blk)
    with pytest.raises(ValueError, match="citations"):
        GroundedAnswer(
            question="q",
            answer="fallback",
            mode="insufficient_context",
            citations=(cit,),
            used_citation_ids=(),
            retrieval_mode=None,
            insufficient_context=True,
        )


def test_citation_preview_truncation() -> None:
    blk = _block(text="x" * 300)
    c = citation_from_block(blk)
    assert len(c.text_preview) == 240
    assert c.text_preview.endswith("\u2026")


def test_grounded_context_block_invalid_citation_id() -> None:
    with pytest.raises(ValueError, match="citation_id"):
        GroundedContextBlock(
            citation_id=0,
            chunk_id=uuid4(),
            document_id=uuid4(),
            source_path=None,
            title=None,
            heading=None,
            rank=1,
            score=None,
            text="hi",
        )
