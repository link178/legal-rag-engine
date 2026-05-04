"""Retrieval evaluation baseline (Phase 5.5): golden JSONL, Hit@k, MRR, JSON/Markdown reports."""

from __future__ import annotations

from app.evaluation.cli import (
    evaluation_item_to_dict,
    evaluation_summary_to_dict,
    main,
    write_markdown_report,
)
from app.evaluation.metrics.retrieval import (
    branch_score,
    compute_hit_at_k,
    compute_reciprocal_rank,
    evaluate_question,
    extracted_lists_from_hits,
    summarize_retrieval_evaluation,
)
from app.evaluation.models import (
    RetrievalEvaluationItem,
    RetrievalEvaluationSummary,
    RetrievalGoldenQuestion,
)
from app.evaluation.runners.retrieval import RetrievalEvaluationRunner, load_golden_questions

__all__ = [
    "RetrievalEvaluationItem",
    "RetrievalEvaluationRunner",
    "RetrievalEvaluationSummary",
    "RetrievalGoldenQuestion",
    "branch_score",
    "compute_hit_at_k",
    "compute_reciprocal_rank",
    "evaluate_question",
    "evaluation_item_to_dict",
    "evaluation_summary_to_dict",
    "extracted_lists_from_hits",
    "load_golden_questions",
    "main",
    "summarize_retrieval_evaluation",
    "write_markdown_report",
]
