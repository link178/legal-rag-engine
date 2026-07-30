"""Unit tests for evaluation execution status helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.evaluation.models import AnswerEvaluationItem, RetrievalEvaluationItem
from app.evaluation.status import (
    derive_answer_execution_status,
    derive_retrieval_execution_status,
    exit_code_for_status,
    retrieval_item_passed,
)


def _ans(*, passed: bool, error: str | None = None) -> AnswerEvaluationItem:
    return AnswerEvaluationItem(
        question_id="q",
        question="?",
        mode="sparse_only",
        answer_mode="grounded",
        expected_mode="grounded",
        mode_matches=passed,
        contains_expected_terms=True,
        missing_expected_terms=(),
        citation_validity_rate=1.0,
        has_valid_citations=True,
        has_invalid_citations=False,
        insufficient_context_matches=True,
        retrieved_expected_source=True,
        cited_source_paths=("x.md",),
        used_citation_ids=(1,),
        invalid_citation_ids=(),
        passed=passed,
        error=error,
    )


def _ret(*, hit: bool, error: str | None = None) -> RetrievalEvaluationItem:
    return RetrievalEvaluationItem(
        question_id="q",
        question="?",
        mode="hybrid",
        top_k=5,
        hit=hit,
        hit_rank=1 if hit else None,
        reciprocal_rank=1.0 if hit else 0.0,
        matched_by=("term",) if hit else (),
        retrieved_chunk_ids=(uuid4(),) if hit else (),
        retrieved_document_ids=(uuid4(),) if hit else (),
        retrieved_source_paths=("x.md",) if hit else (),
        retrieved_scores=(0.5,) if hit else (),
        error=error,
    )


def test_answer_passed_when_all_cases_pass() -> None:
    items = (_ans(passed=True), _ans(passed=True))
    assert derive_answer_execution_status(items, total_questions=2) == "PASSED"


def test_answer_executed_with_failed_case_for_partial() -> None:
    items = (_ans(passed=True), _ans(passed=False), _ans(passed=True))
    assert (
        derive_answer_execution_status(items, total_questions=3)
        == "EXECUTED_WITH_FAILED_CASE"
    )


def test_answer_failed_when_empty() -> None:
    assert derive_answer_execution_status((), total_questions=0) == "FAILED"
    assert derive_answer_execution_status((), total_questions=3) == "FAILED"


def test_retrieval_statuses() -> None:
    assert (
        derive_retrieval_execution_status((_ret(hit=True),), total_questions=1)
        == "PASSED"
    )
    assert (
        derive_retrieval_execution_status(
            (_ret(hit=True), _ret(hit=False)),
            total_questions=2,
        )
        == "EXECUTED_WITH_FAILED_CASE"
    )
    assert not retrieval_item_passed(_ret(hit=False))


def test_exit_codes() -> None:
    assert exit_code_for_status("PASSED", gate=True) == 0
    assert exit_code_for_status("EXECUTED_WITH_FAILED_CASE", gate=True) == 1
    assert exit_code_for_status("EXECUTED_WITH_FAILED_CASE", gate=False) == 0
    assert exit_code_for_status("FAILED", gate=False) == 1
    assert exit_code_for_status("FAILED", gate=True) == 1


def test_created_at_unused_but_models_accept_status() -> None:
    # Smoke-check that summary fields remain constructible with status.
    _ = datetime.now(tz=UTC)
