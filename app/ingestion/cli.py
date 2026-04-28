"""Operator CLI: ingest a single file; optional ``--persist`` to Postgres."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.core.config import get_settings
from app.ingestion.runner import ingest_file_persisted
from app.ingestion.services import default_ingestion_service


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest one .txt or Markdown file into Document.")
    parser.add_argument("path", type=Path, help="File path to ingest")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist document and processing run to PostgreSQL (requires DATABASE_URL / DB up)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print document summary as JSON (non-persist path only adds file fields)",
    )
    args = parser.parse_args(argv)

    service = default_ingestion_service()

    if args.persist:
        settings = get_settings()
        result = ingest_file_persisted(args.path, service, database_url=settings.database_url)
        doc_id = None
        if result.document and result.document.id:
            doc_id = str(result.document.id)
        run_id = None
        if result.run and result.run.id:
            run_id = str(result.run.id)
        out = {
            "persisted": True,
            "created": result.created,
            "skipped": result.skipped,
            "error": result.error,
            "document_id": doc_id,
            "run_id": run_id,
            "run_status": result.run.status if result.run else None,
        }
        print(json.dumps(out, indent=2))
        return 1 if result.error else 0

    try:
        doc = service.ingest_file(args.path)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.as_json:
        payload = {
            "source_path": doc.source_path,
            "source_type": doc.source_type,
            "checksum": doc.checksum,
            "title": doc.title,
            "metadata": doc.metadata,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"source_path: {doc.source_path}")
        print(f"source_type: {doc.source_type}")
        print(f"checksum: {doc.checksum}")
        print(f"title: {doc.title!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
