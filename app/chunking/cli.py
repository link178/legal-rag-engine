"""Operator CLI: chunk a persisted document by UUID."""

from __future__ import annotations

import argparse
import json
import sys
from uuid import UUID

from app.chunking.models import ChunkingConfig
from app.chunking.runner import chunk_document_persisted
from app.core.config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chunk a persisted document by id (requires Postgres + migrations)."
    )
    parser.add_argument(
        "document_id",
        type=str,
        help="UUID of the document row in PostgreSQL",
    )
    parser.add_argument(
        "--strategy",
        choices=("fixed_size", "structure_aware"),
        required=True,
        help="Chunking strategy name",
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--min-chunk-chars", type=int, default=80)
    parser.add_argument(
        "--preserve-headings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For structure_aware: split on Markdown headings when True",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print summary as JSON",
    )
    args = parser.parse_args(argv)

    try:
        doc_uuid = UUID(args.document_id)
    except ValueError:
        print("error: document_id must be a valid UUID", file=sys.stderr)
        return 1

    config = ChunkingConfig(
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        min_chunk_chars=args.min_chunk_chars,
        preserve_headings=args.preserve_headings,
    )

    settings = get_settings()
    result = chunk_document_persisted(doc_uuid, config, database_url=settings.database_url)

    chunk_ids = [str(x) for x in result.chunk_ids]
    run_id = str(result.run.id) if result.run and result.run.id else None

    payload = {
        "persisted": True,
        "document_id": str(doc_uuid),
        "strategy": result.strategy,
        "created_chunks": result.created_chunks,
        "skipped_existing": result.skipped_existing,
        "chunk_ids": chunk_ids,
        "run_id": run_id,
        "run_status": result.run.status if result.run else None,
        "error": result.error,
    }

    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"document_id: {doc_uuid}")
        print(f"strategy: {result.strategy}")
        print(f"created_chunks: {result.created_chunks}")
        print(f"skipped_existing: {result.skipped_existing}")
        print(f"chunk_row_ids: {len(chunk_ids)}")
        if run_id:
            print(f"run_id: {run_id}")
        if result.error:
            print(f"error: {result.error}")

    return 1 if result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
