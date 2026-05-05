"""Operator CLI: import a legal corpus directory (optional Postgres persistence)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.ingestion.adapters.legal_corpus.importer import LegalCorpusImporter
from app.ingestion.adapters.legal_corpus.models import (
    LegalCorpusImportItem,
    LegalCorpusImportSummary,
)
from app.ingestion.services import default_ingestion_service


def item_to_dict(item: LegalCorpusImportItem) -> dict[str, Any]:
    return {
        "document_id": item.document_id,
        "error": item.error,
        "metadata": dict(item.metadata),
        "relative_path": item.relative_path,
        "source_path": item.source_path,
        "status": item.status,
        "title": item.title,
        "warnings": list(item.warnings),
    }


def summary_to_dict(summary: LegalCorpusImportSummary) -> dict[str, Any]:
    return {
        "corpus_name": summary.corpus_name,
        "corpus_path": summary.corpus_path,
        "corpus_run_id": summary.corpus_run_id,
        "discovered_count": summary.discovered_count,
        "failed_count": summary.failed_count,
        "imported_count": summary.imported_count,
        "items": [item_to_dict(i) for i in summary.items],
        "reused_count": summary.reused_count,
    }


def render_markdown(summary: LegalCorpusImportSummary) -> str:
    lines = [
        "# Legal Corpus Import Report",
        "",
        f"- Corpus path: `{summary.corpus_path}`",
        f"- Corpus name: `{summary.corpus_name}`",
        f"- Umbrella run ID: `{summary.corpus_run_id or ''}`",
        f"- Discovered: {summary.discovered_count}",
        f"- Imported: {summary.imported_count}",
        f"- Reused: {summary.reused_count}",
        f"- Failed: {summary.failed_count}",
        "",
        "## Items",
        "",
        "| Status | Relative path | Document ID | Title | Error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in summary.items:
        title = (item.title or "").replace("|", "\\|")
        err = (item.error or "").replace("|", "\\|")
        lines.append(
            f"| {item.status} | {item.relative_path} | {item.document_id or ''} | {title} | {err} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import supported files from a legal corpus directory (local path only).",
    )
    parser.add_argument(
        "corpus_path",
        type=Path,
        help="Root directory of the corpus",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist documents via ingest_file_persisted (Postgres required)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print import summary as JSON",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=None,
        help="Write a Markdown report to this path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Import at most N files (after deterministic discovery order)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failed file (marks umbrella run failed when persisting)",
    )
    args = parser.parse_args(argv)

    root = args.corpus_path.expanduser().resolve()
    if not root.exists():
        print(f"error: path does not exist: {args.corpus_path}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"error: not a directory: {args.corpus_path}", file=sys.stderr)
        return 1

    database_url: str | None = None
    if args.persist:
        database_url = get_settings().database_url

    service = default_ingestion_service()
    importer = LegalCorpusImporter(
        service,
        database_url=database_url,
        fail_fast=args.fail_fast,
        limit=args.limit,
    )

    try:
        summary = importer.import_corpus(root, persist=args.persist)
    except Exception as e:  # noqa: BLE001
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if args.markdown_report is not None:
        args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_report.write_text(render_markdown(summary), encoding="utf-8")

    if args.as_json:
        print(json.dumps(summary_to_dict(summary), indent=2, sort_keys=True))

    if args.persist and summary.discovered_count == 0:
        return 1
    if args.fail_fast and summary.failed_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
