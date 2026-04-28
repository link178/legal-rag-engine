"""Sparse / lexical indexing helpers for Phase 4A term-frequency representation."""

from __future__ import annotations

from app.indexing.sparse.document_terms import extract_sparse_terms
from app.indexing.sparse.tokenizer import tokenize

__all__ = ["extract_sparse_terms", "tokenize"]
