"""Citation id extraction from model output."""

from __future__ import annotations

import pytest
from app.generation.citations import extract_cited_ids, verify_citations


def test_order_preserving_dedupe() -> None:
    assert extract_cited_ids("See [1] and also [2] then [1] again.") == (1, 2)


def test_multiple_in_sequence() -> None:
    assert extract_cited_ids("Ref [1][2][3].") == (1, 2, 3)


def test_ignores_non_digit_brackets() -> None:
    assert extract_cited_ids("[abc] [1]") == (1,)


def test_verify_citations_stub() -> None:
    with pytest.raises(NotImplementedError):
        verify_citations("", [])
