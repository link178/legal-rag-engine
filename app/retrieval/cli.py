"""Operator CLI: hybrid / dense / sparse retrieval (requires Postgres + migrations)."""

from __future__ import annotations

import argparse
import json
import sys
from uuid import UUID

from app.core.config import get_settings
from app.retrieval.dense import DenseRetriever
from app.retrieval.errors import ManifestNotFoundError, RetrievalError
from app.retrieval.filter_cli import add_metadata_filter_flags, metadata_filter_from_args
from app.retrieval.manifest import resolve_manifest_record
from app.retrieval.models import RetrievalConfig
from app.retrieval.orchestrator import RetrievalOrchestrator
from app.retrieval.sparse import SparseRetriever
from app.storage.postgres.session import session_scope

_PREVIEW_LEN = 240


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve chunks for a query (Phase 5: dense pgvector, sparse terms, hybrid RRF). "
            "Requires DATABASE_URL and alembic upgrade head; run indexing CLI first."
        ),
    )
    parser.add_argument("query", type=str, help="Search query text")
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
        help="Use this manifest UUID (must match embeddings for dense path)",
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
    add_metadata_filter_flags(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print JSON result",
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
    meta_f = metadata_filter_from_args(args)

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
            metadata_filter=meta_f,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        with session_scope(settings.database_url) as session:
            if cfg.mode == "sparse_only":
                mf = resolve_manifest_record(session, cfg, for_dense=False)
            else:
                mf = resolve_manifest_record(session, cfg, for_dense=True)
            if cfg.mode == "hybrid" and not mf.include_sparse:
                print(
                    "error: hybrid mode requires a manifest with sparse indexing "
                    "(re-index without --no-sparse)",
                    file=sys.stderr,
                )
                return 1
            orch = RetrievalOrchestrator(
                DenseRetriever(session),
                SparseRetriever(session),
            )
            res = orch.retrieve(args.query, cfg, mf)
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
        payload = {
            "query": res.query,
            "mode": res.mode,
            "top_k": cfg.top_k,
            "manifest_id": str(res.index_manifest_id) if res.index_manifest_id else None,
            "manifest_hash": res.manifest_hash,
            "embedding_provider": res.embedding_provider,
            "embedding_model": res.embedding_model,
            "embedding_dimensions": res.embedding_dimensions,
            "applied_metadata_filter": (
                dict(cfg.metadata_filter.as_dict()) if cfg.metadata_filter else None
            ),
            "total_results": len(res.results),
            "results": [
                {
                    "rank": r.rank_position,
                    "chunk_id": str(r.chunk_id),
                    "document_id": str(r.document_id),
                    "source_path": r.source_path,
                    "title": r.title,
                    "heading": r.heading,
                    "chunk_index": r.chunk_index,
                    "chunking_strategy": r.chunking_strategy,
                    "dense_score": r.dense_score,
                    "sparse_score": r.sparse_score,
                    "rrf_score": r.rrf_score,
                    "retrieval_sources": list(r.retrieval_sources),
                    "text_preview": (r.text[:_PREVIEW_LEN] if r.text else ""),
                }
                for r in res.results
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"results: {len(res.results)} (manifest {res.index_manifest_id})")
        for r in res.results:
            print(
                f"  {r.rank_position}. {r.chunk_id} score_dense={r.dense_score} "
                f"sparse={r.sparse_score} rrf={r.rrf_score} {r.source_path or ''}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
