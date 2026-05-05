"""RetrievalMetadataFilter (Phase 14)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from app.retrieval.models import RetrievalMetadataFilter


def test_is_empty_all_none() -> None:
    f = RetrievalMetadataFilter()
    assert f.is_empty() is True
    assert f.as_dict() == {}


def test_is_empty_whitespace_only() -> None:
    f = RetrievalMetadataFilter(jurisdiction="  ", language="")
    assert f.is_empty() is True


def test_partial_fill_as_dict() -> None:
    f = RetrievalMetadataFilter(jurisdiction="eu", legal_document_type="regulation")
    assert f.is_empty() is False
    assert f.as_dict() == {
        "jurisdiction": "eu",
        "legal_document_type": "regulation",
    }


def test_as_dict_strips_values() -> None:
    f = RetrievalMetadataFilter(corpus_name="  legalize_sample  ")
    assert f.as_dict() == {"corpus_name": "legalize_sample"}


def test_frozen() -> None:
    f = RetrievalMetadataFilter(jurisdiction="es")
    with pytest.raises(FrozenInstanceError):
        f.jurisdiction = "eu"


def test_all_supported_fields() -> None:
    f = RetrievalMetadataFilter(
        corpus_name="a",
        corpus_adapter="b",
        source_family="c",
        jurisdiction="d",
        legal_document_type="e",
        language="f",
        canonical_id="g",
    )
    assert f.as_dict() == {
        "corpus_name": "a",
        "corpus_adapter": "b",
        "source_family": "c",
        "jurisdiction": "d",
        "legal_document_type": "e",
        "language": "f",
        "canonical_id": "g",
    }
