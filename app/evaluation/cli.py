"""Operator CLI: retrieval evaluation (Phase 5.5) + answer evaluation (Phase 11)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

from app.core.config import get_settings
from app.evaluation.models import (
    AnswerEvaluationItem,
    AnswerEvaluationSummary,
    RetrievalEvaluationItem,
    RetrievalEvaluationSummary,
)
from app.evaluation.runners.answer import AnswerEvaluationRunner
from app.evaluation.runners.retrieval import RetrievalEvaluationRunner
from app.evaluation.status import exit_code_for_status
from app.generation.context import ContextBuilder
from app.generation.providers.mock import MockGenerationProvider
from app.retrieval.errors import ManifestNotFoundError, RetrievalError
from app.retrieval.filter_cli import add_metadata_filter_flags, metadata_filter_from_args
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
            "execution_status": summary.execution_status,
        },
        "items": [evaluation_item_to_dict(i) for i in summary.items],
    }


def evaluation_answer_item_to_dict(item: AnswerEvaluationItem) -> dict:
    return {
        "question_id": item.question_id,
        "question": item.question,
        "mode": item.mode,
        "answer_mode": item.answer_mode,
        "expected_mode": item.expected_mode,
        "mode_matches": item.mode_matches,
        "contains_expected_terms": item.contains_expected_terms,
        "missing_expected_terms": list(item.missing_expected_terms),
        "citation_validity_rate": item.citation_validity_rate,
        "has_valid_citations": item.has_valid_citations,
        "has_invalid_citations": item.has_invalid_citations,
        "insufficient_context_matches": item.insufficient_context_matches,
        "retrieved_expected_source": item.retrieved_expected_source,
        "cited_source_paths": list(item.cited_source_paths),
        "used_citation_ids": list(item.used_citation_ids),
        "invalid_citation_ids": list(item.invalid_citation_ids),
        "passed": item.passed,
        "error": item.error,
    }


def evaluation_answer_summary_to_dict(summary: AnswerEvaluationSummary) -> dict:
    return {
        "schema_version": summary.schema_version,
        "created_at": summary.created_at.isoformat(),
        "config": dict(summary.config),
        "manifest": dict(summary.manifest),
        "summary": {
            "total_questions": summary.total_questions,
            "answered_questions": summary.answered_questions,
            "errored_questions": summary.errored_questions,
            "pass_rate": summary.pass_rate,
            "mode_accuracy": summary.mode_accuracy,
            "expected_terms_accuracy": summary.expected_terms_accuracy,
            "citation_validity_rate_avg": summary.citation_validity_rate_avg,
            "insufficient_context_accuracy": summary.insufficient_context_accuracy,
            "retrieved_expected_source_rate": summary.retrieved_expected_source_rate,
            "retrieved_expected_source_notes": summary.retrieved_expected_source_notes,
            "execution_status": summary.execution_status,
        },
        "items": [evaluation_answer_item_to_dict(i) for i in summary.items],
    }


def write_markdown_report(summary: RetrievalEvaluationSummary, path: Path) -> None:
    lines = [
        "# Retrieval evaluation report",
        "",
        f"- mode: {summary.config.get('mode')}",
        f"- top_k: {summary.config.get('top_k')}",
        f"- manifest_id: {summary.manifest.get('id')}",
        f"- execution_status: {summary.execution_status}",
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


def write_answer_markdown_report(summary: AnswerEvaluationSummary, path: Path) -> None:
    lines = [
        "# Answer evaluation report",
        "",
        f"- mode: {summary.config.get('mode')}",
        f"- top_k: {summary.config.get('top_k')}",
        f"- provider: {summary.config.get('provider')}",
        f"- manifest_id: {summary.manifest.get('id')}",
        f"- execution_status: {summary.execution_status}",
        f"- pass_rate: {summary.pass_rate:.4f}  ({summary.answered_questions} answered)",
        f"- mode_accuracy: {summary.mode_accuracy:.4f}",
        f"- expected_terms_accuracy: {summary.expected_terms_accuracy:.4f}",
        f"- citation_validity_rate_avg: {summary.citation_validity_rate_avg:.4f}",
        f"- insufficient_context_accuracy: {summary.insufficient_context_accuracy:.4f}",
        f"- retrieved_expected_source_rate: {summary.retrieved_expected_source_rate:.4f}",
        f"- errors: {summary.errored_questions}",
        "",
        "## Per-question",
        "",
    ]
    for it in summary.items:
        status = "PASS" if it.passed else ("ERR" if it.error else "FAIL")
        err = f" err={it.error!r}" if it.error else ""
        lines.append(
            f"- {it.question_id}  {status}  answer_mode={it.answer_mode}  "
            f"missing_terms={list(it.missing_expected_terms)}  "
            f"cited={list(it.cited_source_paths)}{err}"
        )
    lines.extend(["", "## Failures", ""])
    failures = [it for it in summary.items if not it.passed]
    if not failures:
        lines.append("- (none)")
    else:
        for it in failures:
            parts = [
                f"- {it.question_id}:",
                f"  mode_match={it.mode_matches}",
                f"  terms_ok={it.contains_expected_terms}",
                f"  missing_terms={list(it.missing_expected_terms)}",
                f"  citation_validity={it.citation_validity_rate:.2f}",
                f"  invalid_ids={list(it.invalid_citation_ids)}",
                f"  insufficient_ok={it.insufficient_context_matches}",
                f"  source_ok={it.retrieved_expected_source!r}",
            ]
            if it.error:
                parts.append(f"  error={it.error!r}")
            lines.extend(parts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _retrieval_config_from_args(
    args: argparse.Namespace, settings
) -> tuple[RetrievalConfig | None, int]:
    mode = args.mode if args.mode is not None else settings.retrieval_mode

    mid: UUID | None = None
    if args.index_manifest_id:
        try:
            mid = UUID(args.index_manifest_id.strip())
        except ValueError:
            print("error: --index-manifest-id must be a valid UUID", file=sys.stderr)
            return None, 1

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
    meta_filt = metadata_filter_from_args(args)

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
            metadata_filter=meta_filt,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return None, 1
    return cfg, 0


def _build_retrieval_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--no-gate",
        action="store_true",
        dest="no_gate",
        help=(
            "Informational mode: exit 0 when evaluation executed even if cases "
            "failed; infrastructure failures still exit non-zero"
        ),
    )
    add_metadata_filter_flags(parser)
    return parser


def _run_retrieval(argv: list[str]) -> int:
    parser = _build_retrieval_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    cfg, rc = _retrieval_config_from_args(args, settings)
    if rc != 0 or cfg is None:
        return rc

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
            f"eval: questions={summary.total_questions} "
            f"execution_status={summary.execution_status} "
            f"hit_rate={summary.hit_rate:.4f} "
            f"mrr={summary.mrr:.4f} errors={summary.errored_questions} "
            f"manifest={summary.manifest.get('id')}"
        )

    return exit_code_for_status(summary.execution_status, gate=not args.no_gate)


def _build_answer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate grounded answers against a golden JSONL file (Phase 11). "
            "Requires DATABASE_URL, alembic head, ingest + chunk + indexing first; "
            "uses mock generation only."
        ),
    )
    parser.add_argument(
        "golden_file",
        type=str,
        help="Path to JSONL file (one answer golden question per line)",
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
        help="Pin this manifest UUID",
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
        help="Generation provider (Phase 11: only mock supported)",
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
    parser.add_argument(
        "--no-gate",
        action="store_true",
        dest="no_gate",
        help=(
            "Informational mode: exit 0 when evaluation executed even if cases "
            "failed; infrastructure failures still exit non-zero"
        ),
    )
    add_metadata_filter_flags(parser)
    return parser


def _run_answer(argv: list[str]) -> int:
    parser = _build_answer_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    cfg, rc = _retrieval_config_from_args(args, settings)
    if rc != 0 or cfg is None:
        return rc

    raw_provider = (
        args.provider if args.provider is not None else settings.generation_provider
    )
    provider_name = str(raw_provider or "").strip().lower()
    if provider_name != "mock":
        print(
            "error: Phase 11 answer evaluation only supports generation provider "
            f"'mock'; got {raw_provider!r}",
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

    golden_path = Path(args.golden_file)
    try:
        with session_scope(settings.database_url) as session:
            runner = AnswerEvaluationRunner.from_session(
                session,
                cfg,
                context_builder=ctx_builder,
                provider=mock,
            )
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

    payload = evaluation_answer_summary_to_dict(summary)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.markdown_output:
        write_answer_markdown_report(summary, Path(args.markdown_output))

    if args.as_json:
        print(json.dumps(payload, indent=2))

    if not args.as_json and not args.output:
        print(
            f"answer_eval: questions={summary.total_questions} "
            f"execution_status={summary.execution_status} "
            f"pass_rate={summary.pass_rate:.4f} errors={summary.errored_questions} "
            f"manifest={summary.manifest.get('id')}"
        )

    return exit_code_for_status(summary.execution_status, gate=not args.no_gate)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return _run_retrieval(argv)
    cmd = argv[0]
    if cmd == "retrieval":
        return _run_retrieval(argv[1:])
    if cmd == "answer":
        return _run_answer(argv[1:])
    return _run_retrieval(argv)


if __name__ == "__main__":
    raise SystemExit(main())
