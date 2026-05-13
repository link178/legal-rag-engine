"""Retrieval (Phase 5.5) + answer (Phase 11) evaluation."""

from __future__ import annotations

from app.evaluation.cli import (
    evaluation_answer_item_to_dict,
    evaluation_answer_summary_to_dict,
    evaluation_item_to_dict,
    evaluation_summary_to_dict,
    main,
    write_answer_markdown_report,
    write_markdown_report,
)
from app.evaluation.metrics.answer import (
    compute_pass,
    evaluate_answer_question,
    summarize_answer_evaluation,
    terms_match_contains,
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
    AnswerEvaluationItem,
    AnswerEvaluationSummary,
    AnswerGoldenQuestion,
    RetrievalEvaluationItem,
    RetrievalEvaluationSummary,
    RetrievalGoldenQuestion,
)
from app.evaluation.runners.answer import AnswerEvaluationRunner, load_answer_golden
from app.evaluation.runners.retrieval import RetrievalEvaluationRunner, load_golden_questions

__all__ = [
    "AnswerEvaluationItem",
    "AnswerEvaluationRunner",
    "AnswerEvaluationSummary",
    "AnswerGoldenQuestion",
    "RetrievalEvaluationItem",
    "RetrievalEvaluationRunner",
    "RetrievalEvaluationSummary",
    "RetrievalGoldenQuestion",
    "branch_score",
    "compute_hit_at_k",
    "compute_pass",
    "compute_reciprocal_rank",
    "evaluate_answer_question",
    "evaluate_question",
    "evaluation_answer_item_to_dict",
    "evaluation_answer_summary_to_dict",
    "evaluation_item_to_dict",
    "evaluation_summary_to_dict",
    "extracted_lists_from_hits",
    "load_answer_golden",
    "load_golden_questions",
    "main",
    "summarize_answer_evaluation",
    "summarize_retrieval_evaluation",
    "terms_match_contains",
    "write_answer_markdown_report",
    "write_markdown_report",
]
