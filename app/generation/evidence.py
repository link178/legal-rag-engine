"""Deterministic lexical evidence sufficiency for grounded answering.

Used to refuse ``grounded`` when retrieved context does not materially support
the question. This is not NLI / entailment — only stopword-aware term overlap
over block text plus source/title/heading fields.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.generation.models import GroundedContextBlock
from app.indexing.sparse.tokenizer import tokenize

# Small fixed English stopword set for offline deterministic checks.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "doing",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "into",
        "about",
        "over",
        "after",
        "before",
        "between",
        "through",
        "during",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "than",
        "too",
        "very",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
        "shall",
        "will",
        "just",
        "also",
        "only",
        "own",
        "same",
        "so",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "his",
        "her",
        "not",
        "no",
        "nor",
        "according",
        "per",
        "via",
    }
)

_MIN_CONTENT_TERM_LEN = 3


def content_terms(text: str) -> tuple[str, ...]:
    """Return unique content tokens (order-preserving) from ``text``."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokenize(text):
        if len(tok) < _MIN_CONTENT_TERM_LEN:
            continue
        if tok in _STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return tuple(out)


def _block_search_text(block: GroundedContextBlock) -> str:
    parts: list[str] = []
    for part in (block.text, block.source_path, block.title, block.heading):
        if part is not None and str(part).strip():
            parts.append(str(part))
    return " ".join(parts)


def matched_content_terms(
    question: str,
    blocks: Sequence[GroundedContextBlock],
) -> tuple[str, ...]:
    """Content query terms that appear in any block field."""
    q_terms = content_terms(question)
    if not q_terms or not blocks:
        return ()
    corpus = " ".join(_block_search_text(b) for b in blocks)
    corpus_terms = set(tokenize(corpus))
    return tuple(t for t in q_terms if t in corpus_terms)


def evidence_is_sufficient(
    question: str,
    blocks: Sequence[GroundedContextBlock] | Iterable[GroundedContextBlock],
) -> bool:
    """Return True when context lexically supports answering ``question``.

    Rule: matched content terms >= ``min(2, len(query_content_terms))``.
    Empty query content terms or empty blocks are never sufficient.
    """
    block_list = tuple(blocks)
    q_terms = content_terms(question)
    if not q_terms or not block_list:
        return False
    matched = matched_content_terms(question, block_list)
    need = min(2, len(q_terms))
    return len(matched) >= need
