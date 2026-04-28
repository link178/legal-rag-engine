"""Sparse tokenizer and term extraction tests."""

from __future__ import annotations

from app.indexing.sparse.document_terms import extract_sparse_terms
from app.indexing.sparse.tokenizer import tokenize


def test_tokenize_lowercase_and_words() -> None:
    assert tokenize("Hello WORLD") == ["hello", "world"]


def test_tokenize_preserves_numbers_in_token() -> None:
    assert tokenize("Section 12A") == ["section", "12a"]


def test_extract_sparse_term_frequencies() -> None:
    d = extract_sparse_terms("cat cat dog")
    assert d == {"cat": 2, "dog": 1}
