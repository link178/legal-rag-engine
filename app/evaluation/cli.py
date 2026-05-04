"""Operator CLI: retrieval-only evaluation over a golden JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

from app.core.config import get_settings
from app.evaluation.models import RetrievalEvaluationItem, RetrievalEvaluationSummary
from app.evaluation.runners.retrieval import RetrievalEvaluationRunner
from app.retrieval.errors import ManifestNotFoundError, RetrievalError
from app.retrieval.models import RetrievalConfig
from app.storage.postgres.session import session_scope


def evaluation_item_to_dict(item: RetrievalEvaluationItem) -> dict:
    return {
        "question_id": item.question_id,
        "question": item.question,
        "mode": item.mode,
        "top_k": item.top_k,
        "hit": item.hit,
        "hit_rank": item.hit_rank,
        "reciprocal_rank": item.reciprocal_rank,
        "matched_by": list(item.matched_by),
        "retrieved_chunk_ids": [str(u) for u in item.retrieved_chunk_ids],
        "retrieved_document_ids": [str(u) for u in item.retrieved_document_ids],
        "retrieved_source_paths": list(item.retrieved_source_paths),
        "retrieved_scores": list(item.retrieved_scores),
        "error": item.error,
    }


def evaluation_summary_to_dict(summary: RetrievalEvaluationSummary) -> dict:
    return {
        "schema_version": summary.schema_version,
        "created_at": summary.created_at.isoformat(),
        "config": dict(summary.config),
        "manifest": dict(summary.manifest),
        "summary": {
            "total_questions": summary.total_questions,
            "answered_questions": summary.answered_questions,
            "errored_questions": summary.errored_questions,
            "hit_rate": summary.hit_rate,
            "mrr": summary.mrr,
            "hit_at_k_notes": summary.hit_at_k_notes,
        },
        "items": [evaluation_item_to_dict(i) for i in summary.items],
    }


def write_markdown_report(summary: RetrievalEvaluationSummary, path: Path) -> None:
    lines = [
        "# Retrieval evaluation report",
        "",
        f"- mode: {summary.config.get('mode')}",
        f"- top_k: {summary.config.get('top_k')}",
        f"- manifest_id: {summary.manifest.get('id')}",
        f"- hit_rate: {summary.hit_rate:.4f}  ({summary.answered_questions} answered)",
        f"- MRR: {summary.mrr:.4f}",
        f"- errors: {summary.errored_questions}",
        "",
        "## Per-question",
        "",
    ]
    for it in summary.items:
        status = "HIT" if it.hit else ("ERR" if it.error else "MISS")
        via = ",".join(it.matched_by) if it.matched_by else "-"
        rank = str(it.hit_rank) if it.hit_rank is not None else "-"
        top_path = it.retrieved_source_paths[0] if it.retrieved_source_paths else "-"
        err = f" err={it.error!r}" if it.error else ""
        lines.append(
            f"- {it.question_id}  {status}  rank={rank}  via={via}  source={top_path}{err}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval against a golden JSONL file (Phase 5.5). "
            "Requires DATABASE_URL, alembic head, ingest + chunk + indexing CLI first."
        ),
    )
    parser.add_argument(
        "golden_file",
        type=str,
        help="Path to JSONL file (one golden question object per line)",
    )
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
        dest="index_manifest_id",
        help="Use this manifest UUID",
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
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print full JSON report to stdout",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write full JSON report to this path",
    )
    parser.add_argument(
        "--markdown-output",
        type=str,
        default=None,
        help="Write Markdown summary to this path",
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

    golden_path = Path(args.golden_file)
    try:
        with session_scope(settings.database_url) as session:
            runner = RetrievalEvaluationRunner.from_session(session, cfg)
            summary = runner.run_file(golden_path, top_k=args.top_k)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: golden file: {e}", file=sys.stderr)
        return 1
    except ManifestNotFoundError as e:
        print(
            f"error: {e}. Ingest, chunk, then run: python -m app.indexing.cli",
            file=sys.stderr,
        )
        return 1
    except RetrievalError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    payload = evaluation_summary_to_dict(summary)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.markdown_output:
        write_markdown_report(summary, Path(args.markdown_output))

    if args.as_json:
        print(json.dumps(payload, indent=2))

    if not args.as_json and not args.output:
        print(
            f"eval: questions={summary.total_questions} hit_rate={summary.hit_rate:.4f} "
            f"mrr={summary.mrr:.4f} errors={summary.errored_questions} "
            f"manifest={summary.manifest.get('id')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
