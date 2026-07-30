"""Evaluation execution status: quality vs infrastructure outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from app.evaluation.models import AnswerEvaluationItem, RetrievalEvaluationItem

EvaluationExecutionStatus = Literal[
    "PASSED",
    "EXECUTED_WITH_FAILED_CASE",
    "FAILED",
]

SUPPORTED_EVALUATION_EXECUTION_STATUSES: tuple[str, ...] = (
    "PASSED",
    "EXECUTED_WITH_FAILED_CASE",
    "FAILED",
)


def retrieval_item_passed(item: RetrievalEvaluationItem) -> bool:
    """A retrieval case passes when it hit with no runtime error."""
    return item.error is None and item.hit is True


def derive_answer_execution_status(
    items: Sequence[AnswerEvaluationItem],
    *,
    total_questions: int,
) -> EvaluationExecutionStatus:
    """Derive answer-eval status after a completed runner invocation."""
    if total_questions <= 0 or len(items) == 0:
        return "FAILED"
    if len(items) != total_questions:
        return "FAILED"
    if all(i.passed for i in items):
        return "PASSED"
    return "EXECUTED_WITH_FAILED_CASE"


def derive_retrieval_execution_status(
    items: Sequence[RetrievalEvaluationItem],
    *,
    total_questions: int,
) -> EvaluationExecutionStatus:
    """Derive retrieval-eval status after a completed runner invocation."""
    if total_questions <= 0 or len(items) == 0:
        return "FAILED"
    if len(items) != total_questions:
        return "FAILED"
    if all(retrieval_item_passed(i) for i in items):
        return "PASSED"
    return "EXECUTED_WITH_FAILED_CASE"


def exit_code_for_status(
    status: str,
    *,
    gate: bool,
) -> int:
    """Map execution status to process exit code.

    When ``gate`` is False (``--no-gate``), a completed run with failed cases
    still exits 0; only ``FAILED`` (unreliable execution) exits non-zero.
    """
    if status == "PASSED":
        return 0
    if status == "EXECUTED_WITH_FAILED_CASE":
        return 0 if not gate else 1
    return 1
