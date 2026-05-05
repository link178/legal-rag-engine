"""Shared argparse flags for metadata filters on retrieval CLIs (Phase 14)."""

from __future__ import annotations

import argparse

from app.retrieval.models import RetrievalMetadataFilter


def add_metadata_filter_flags(parser: argparse.ArgumentParser) -> None:
    """Register optional ``--filter-*`` arguments (exact JSONB match on document metadata)."""
    parser.add_argument(
        "--filter-corpus-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Restrict to documents with metadata corpus_name equal to this value",
    )
    parser.add_argument(
        "--filter-corpus-adapter",
        type=str,
        default=None,
        metavar="NAME",
        help="Restrict to documents with metadata corpus_adapter equal to this value",
    )
    parser.add_argument(
        "--filter-source-family",
        type=str,
        default=None,
        metavar="NAME",
        help="Restrict to documents with metadata source_family equal to this value",
    )
    parser.add_argument(
        "--filter-jurisdiction",
        type=str,
        default=None,
        metavar="CODE",
        help="Restrict to documents with metadata jurisdiction equal to this value",
    )
    parser.add_argument(
        "--filter-legal-document-type",
        type=str,
        default=None,
        metavar="TYPE",
        help="Restrict to documents with metadata legal_document_type equal to this value",
    )
    parser.add_argument(
        "--filter-language",
        type=str,
        default=None,
        metavar="CODE",
        help="Restrict to documents with metadata language equal to this value",
    )
    parser.add_argument(
        "--filter-canonical-id",
        type=str,
        default=None,
        metavar="ID",
        help="Restrict to documents with metadata canonical_id equal to this value",
    )


def metadata_filter_from_args(
    args: argparse.Namespace | object,
) -> RetrievalMetadataFilter | None:
    """Build a filter from CLI args, or ``None`` if no flag was set to a non-empty string."""
    def _s(attr: str) -> str | None:
        raw = getattr(args, attr, None)
        if raw is None:
            return None
        s = str(raw).strip()
        return s or None

    f = RetrievalMetadataFilter(
        corpus_name=_s("filter_corpus_name"),
        corpus_adapter=_s("filter_corpus_adapter"),
        source_family=_s("filter_source_family"),
        jurisdiction=_s("filter_jurisdiction"),
        legal_document_type=_s("filter_legal_document_type"),
        language=_s("filter_language"),
        canonical_id=_s("filter_canonical_id"),
    )
    return None if f.is_empty() else f
