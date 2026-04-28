"""Operator CLI: run persisted indexing (requires Postgres + migrations)."""

from __future__ import annotations

import argparse
import json
import sys

from app.core.config import get_settings
from app.indexing.models import IndexingConfig
from app.indexing.runner import index_chunks_persisted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Index persisted chunks under an index manifest (Phase 4A). "
            "Requires DATABASE_URL and alembic upgrade head."
        ),
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default=None,
        metavar="NAME",
        help="Only chunks with this chunking_strategy; omit for whole corpus table",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default="deterministic_hash",
        help="Phase 4A supports deterministic_hash only",
    )
    parser.add_argument("--embedding-dimensions", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--no-dense",
        action="store_true",
        help="Skip dense embeddings (sparse only)",
    )
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Skip sparse term extraction (dense only)",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Ignore idempotent manifest skip and build a new manifest",
    )
    parser.add_argument(
        "--corpus-version",
        type=str,
        default=None,
        help="Optional corpus label stored on the manifest metadata",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print summary as JSON",
    )
    args = parser.parse_args(argv)

    if args.no_dense and args.no_sparse:
        print("error: cannot disable both dense and sparse", file=sys.stderr)
        return 1

    strat = (args.chunking_strategy or "").strip() or None

    cfg = IndexingConfig(
        embedding_provider=args.embedding_provider,
        embedding_dimensions=args.embedding_dimensions,
        chunking_strategy=strat,
        batch_size=args.batch_size,
        include_dense=not args.no_dense,
        include_sparse=not args.no_sparse,
        force_reindex=args.force_reindex,
    )

    settings = get_settings()
    res = index_chunks_persisted(
        cfg,
        database_url=settings.database_url,
        corpus_version=args.corpus_version,
    )

    mid = None
    if res.manifest and res.manifest.id:
        mid = str(res.manifest.id)
    rid = None
    if res.run and res.run.id:
        rid = str(res.run.id)

    payload = {
        "persisted": True,
        "manifest_hash": res.manifest.manifest_hash if res.manifest else None,
        "manifest_id": mid,
        "run_id": rid,
        "run_status": res.run.status if res.run else None,
        "skipped_existing": res.skipped_existing,
        "indexed_chunks": len(res.chunk_results),
        "indexed_chunk_success_count": (
            sum(1 for x in res.chunk_results if x.error is None)
            if res.chunk_results
            else (
                res.manifest.indexed_chunk_count
                if res.manifest
                else 0
            )
        ),
        "error": res.error,
    }

    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        if res.error:
            print(f"error: {res.error}", file=sys.stderr)
        elif res.skipped_existing:
            print("skipped_existing: manifest already up to date (use --force-reindex)")
        else:
            print(f"indexed chunks recorded: {payload['indexed_chunks']}")
        if rid:
            print(f"run_id: {rid}")
        if mid:
            print(f"manifest_id: {mid}")

    return 1 if res.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
