"""Operator CLI: grounded answer with retrieval + mock generation (requires Postgres)."""

from __future__ import annotations

import argparse
import json
import sys
from uuid import UUID

from app.core.config import get_settings
from app.generation.answerer import GroundedAnswerer
from app.generation.context import ContextBuilder
from app.generation.models import GroundedAnswer
from app.generation.providers.mock import MockGenerationProvider
from app.retrieval.errors import ManifestNotFoundError, RetrievalError
from app.retrieval.models import RetrievalConfig
from app.storage.postgres.session import session_scope


def answer_to_dict(answer: GroundedAnswer) -> dict:
    """JSON-serialize a ``GroundedAnswer`` (nested dicts/lists only)."""
    return {
        "question": answer.question,
        "answer": answer.answer,
        "mode": answer.mode,
        "insufficient_context": answer.insufficient_context,
        "retrieval_mode": answer.retrieval_mode,
        "citations": [
            {
                "citation_id": c.citation_id,
                "chunk_id": str(c.chunk_id),
                "document_id": str(c.document_id),
                "source_path": c.source_path,
                "title": c.title,
                "heading": c.heading,
                "rank": c.rank,
                "score": c.score,
                "text_preview": c.text_preview,
            }
            for c in answer.citations
        ],
        "used_citation_ids": list(answer.used_citation_ids),
        "metadata": dict(answer.metadata),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Answer a question with retrieval + grounded mock generation (Phase 6). "
            "Requires DATABASE_URL and alembic upgrade head; run indexing CLI first."
        ),
    )
    parser.add_argument("question", type=str, help="User question text")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("dense_only", "sparse_only", "hybrid"),
        default=None,
        help="Override retrieval mode (default: RETRIEVAL_MODE from env)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-top-k", type=int, default=10)
    parser.add_argument("--sparse-top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--index-manifest-id",
        type=str,
        default=None,
        help="Pin this manifest UUID (dense path requires persisted embeddings)",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default=None,
        help="Filter auto-selected manifest (default from EMBEDDING_PROVIDER)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Filter auto-selected manifest (default from EMBEDDING_MODEL)",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=None,
        help="Filter auto-selected manifest (default from EMBEDDING_DIMENSIONS)",
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default=None,
        metavar="NAME",
        help="Filter auto-selected manifest by chunking_strategy",
    )
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--max-chunks", type=int, default=5)
    parser.add_argument("--max-chunk-chars", type=int, default=2000)
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Drop retrieval hits whose branch score is below this threshold",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Generation provider (Phase 6: only mock supported; "
            "default from GENERATION_PROVIDER)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print GroundedAnswer as JSON",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    mode = args.mode if args.mode is not None else settings.retrieval_mode

    mid: UUID | None = None
    if args.index_manifest_id:
        try:
            mid = UUID(args.index_manifest_id.strip())
        except ValueError:
            print("error: --index-manifest-id must be a valid UUID", file=sys.stderr)
            return 1

    raw_provider = (
        args.provider if args.provider is not None else settings.generation_provider
    )
    provider_name = str(raw_provider or "").strip().lower()
    if provider_name != "mock":
        print(
            "error: only generation provider mock is supported in Phase 6; "
            f"got {raw_provider!r}",
            file=sys.stderr,
        )
        return 1

    try:
        ctx_builder = ContextBuilder(
            max_chunks=args.max_chunks,
            max_context_chars=args.max_context_chars,
            max_chunk_chars=args.max_chunk_chars,
            min_score=args.min_score,
        )
    except ValueError as e:
        print(f"error: context builder: {e}", file=sys.stderr)
        return 1

    mock = MockGenerationProvider()

    prov = (
        args.embedding_provider
        if args.embedding_provider is not None
        else settings.embedding_provider
    )
    dims = (
        args.embedding_dimensions
        if args.embedding_dimensions is not None
        else settings.embedding_dimensions
    )
    raw_model = (
        args.embedding_model if args.embedding_model is not None else settings.embedding_model
    )
    model_norm = (raw_model or "").strip() or None
    strat = (args.chunking_strategy or "").strip() or None

    try:
        cfg = RetrievalConfig(
            mode=mode,
            top_k=args.top_k,
            dense_top_k=args.dense_top_k,
            sparse_top_k=args.sparse_top_k,
            rrf_k=args.rrf_k,
            index_manifest_id=mid,
            embedding_provider=prov,
            embedding_model=model_norm,
            embedding_dimensions=dims,
            chunking_strategy=strat,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        with session_scope(settings.database_url) as session:
            ga = GroundedAnswerer.from_session(
                session,
                cfg,
                context_builder=ctx_builder,
                provider=mock,
            )
            out = ga.answer(args.question)
    except ManifestNotFoundError as e:
        print(
            f"error: {e}. Ingest, chunk, then run: python -m app.indexing.cli",
            file=sys.stderr,
        )
        return 1
    except RetrievalError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(answer_to_dict(out), indent=2))
    else:
        print(out.answer)
        print(f"\nmode: {out.mode}  insufficient_context={out.insufficient_context}")
        if out.citations:
            print("citations:")
            for c in out.citations[:5]:
                pv = (
                    (c.text_preview[:80] + "\u2026")
                    if len(c.text_preview) > 80
                    else c.text_preview
                )
                print(f"  [{c.citation_id}] {c.source_path or '—'}  {pv}")
        elif out.used_citation_ids:
            print(f"citation ids cited: {list(out.used_citation_ids)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
