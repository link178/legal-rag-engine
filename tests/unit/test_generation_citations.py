"""Citation id extraction from model output and mechanical verification."""

from __future__ import annotations

from uuid import uuid4

from app.generation.citations import extract_cited_ids, verify_citations
from app.generation.models import GroundedContextBlock


def _ctx_block(*, citation_id: int = 1) -> GroundedContextBlock:
    return GroundedContextBlock(
        citation_id=citation_id,
        chunk_id=uuid4(),
        document_id=uuid4(),
        source_path="p.md",
        title=None,
        heading=None,
        rank=1,
        score=0.1,
        text="snippet",
    )


def test_order_preserving_dedupe() -> None:
    assert extract_cited_ids("See [1] and also [2] then [1] again.") == (1, 2)


def test_multiple_in_sequence() -> None:
    assert extract_cited_ids("Ref [1][2][3].") == (1, 2, 3)


def test_ignores_non_digit_brackets() -> None:
    assert extract_cited_ids("[abc] [1]") == (1,)


def test_verify_all_valid() -> None:
    blocks = (_ctx_block(citation_id=1), _ctx_block(citation_id=2))
    r = verify_citations("... [1] [2]", blocks)
    assert r.valid_citation_ids == (1, 2)
    assert r.invalid_citation_ids == ()
    assert r.citation_validity_rate == 1.0


def test_verify_only_invalid() -> None:
    blocks = (_ctx_block(citation_id=1), _ctx_block(citation_id=2))
    r = verify_citations("... [3]", blocks)
    assert r.valid_citation_ids == ()
    assert r.invalid_citation_ids == (3,)
    assert r.citation_validity_rate == 0.0


def test_verify_mixed_valid_invalid() -> None:
    blocks = (_ctx_block(citation_id=1), _ctx_block(citation_id=2))
    r = verify_citations("... [1] [99]", blocks)
    assert r.valid_citation_ids == (1,)
    assert r.invalid_citation_ids == (99,)
    assert r.citation_validity_rate == 0.5


def test_verify_duplicates() -> None:
    blocks = (_ctx_block(citation_id=1), _ctx_block(citation_id=2))
    r = verify_citations("... [1] [1] [2]", blocks)
    assert r.duplicate_citation_ids == (1,)
    assert r.valid_citation_ids == (1, 2)
    assert r.citation_validity_rate == 1.0


def test_verify_no_citations() -> None:
    blocks = (_ctx_block(citation_id=1),)
    r = verify_citations("text without citations", blocks)
    assert r.has_citations is False
    assert r.citation_validity_rate == 0.0


def test_verify_empty_context_but_cited() -> None:
    r = verify_citations("See [1].", ())
    assert r.invalid_citation_ids == (1,)
    assert r.available_citation_ids == ()


def test_verify_unused() -> None:
    blocks = tuple(_ctx_block(citation_id=i) for i in (1, 2, 3))
    r = verify_citations("... [1]", blocks)
    assert r.unused_citation_ids == (2, 3)
