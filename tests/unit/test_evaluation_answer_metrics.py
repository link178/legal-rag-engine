"""Pure answer evaluation metric functions."""

from __future__ import annotations

from uuid import uuid4

from app.evaluation.metrics.answer import (
    compute_pass,
    evaluate_answer_question,
    summarize_answer_evaluation,
    terms_match_contains,
)
from app.evaluation.models import AnswerEvaluationItem, AnswerGoldenQuestion
from app.generation.models import (
    CitationVerificationResult,
    GroundedAnswer,
    GroundedCitation,
    empty_verification,
)


def _citation(cid: int, preview: str, path: str | None = "p.md") -> GroundedCitation:
    return GroundedCitation(
        citation_id=cid,
        chunk_id=uuid4(),
        document_id=uuid4(),
        source_path=path,
        title=None,
        heading=None,
        rank=cid,
        score=0.5,
        text_preview=preview,
    )


def _grounded_answer(
    *,
    answer: str = "Summary [1].",
    mode: str = "grounded",
    citations: tuple[GroundedCitation, ...] = (),
    used: tuple[int, ...] = (1,),
    cv: CitationVerificationResult | None = None,
) -> GroundedAnswer:
    if cv is None:
        cv = CitationVerificationResult(
            used_citation_ids=used,
            available_citation_ids=(1,),
            valid_citation_ids=used,
            invalid_citation_ids=(),
            unused_citation_ids=tuple(x for x in (1,) if x not in set(used)),
            duplicate_citation_ids=(),
            citation_validity_rate=1.0,
            has_citations=True,
            has_valid_citations=True,
            has_invalid_citations=False,
        )
    insuff = mode == "insufficient_context"
    return GroundedAnswer(
        question="q?",
        answer=answer,
        mode=mode,
        citations=citations,
        used_citation_ids=used if mode == "grounded" else (),
        retrieval_mode="hybrid",
        insufficient_context=insuff,
        citation_verification=cv if mode != "insufficient_context" else empty_verification(),
    )


def test_terms_match_answer_only() -> None:
    ok, miss = terms_match_contains(
        "Hello world sample",
        ("no match here",),
        ("hello", "sample"),
        expected_terms_in="answer",
    )
    assert ok and miss == ()


def test_terms_match_citations_only() -> None:
    ok, miss = terms_match_contains(
        "mock fixed text",
        ("has markdown here",),
        ("markdown",),
        expected_terms_in="citations",
    )
    assert ok and miss == ()


def test_terms_match_both_or() -> None:
    ok, miss = terms_match_contains(
        "only in preview",
        ("alpha beta",),
        ("alpha",),
        expected_terms_in="both",
    )
    assert ok and miss == ()


def test_terms_missing() -> None:
    ok, miss = terms_match_contains("nope", ("nada",), ("x",), expected_terms_in="both")
    assert not ok and miss == ("x",)


def test_compute_pass_expected_mode_none() -> None:
    assert compute_pass(
        error=None,
        answer_mode="partial",
        expected_mode=None,
        mode_matches=True,
        insufficient_context_matches=True,
        contains_expected_terms=True,
        expected_terms=(),
        has_valid_citations=False,
        has_invalid_citations=True,
        retrieved_expected_source=True,
        expected_source_paths=(),
    )


def test_compute_pass_grounded_requires_valid_citations() -> None:
    assert not compute_pass(
        error=None,
        answer_mode="grounded",
        expected_mode="grounded",
        mode_matches=True,
        insufficient_context_matches=True,
        contains_expected_terms=True,
        expected_terms=("a",),
        has_valid_citations=False,
        has_invalid_citations=False,
        retrieved_expected_source=True,
        expected_source_paths=(),
    )


def test_evaluate_intro_style_grounded() -> None:
    g = AnswerGoldenQuestion(
        id="t",
        question="?",
        expected_mode="grounded",
        expected_terms=("markdown",),
        expected_source_paths=("intro.md",),
    )
    ans = _grounded_answer(
        citations=(_citation(1, "Short **Markdown** sample", "intro.md"),),
    )
    item = evaluate_answer_question(g, ans, retrieval_mode="hybrid")
    assert item.passed
    assert item.retrieved_expected_source is True


def test_evaluate_partial_mode_when_expected_grounded() -> None:
    g = AnswerGoldenQuestion(
        id="t",
        question="?",
        expected_mode="grounded",
        expected_terms=(),
        expected_source_paths=(),
    )
    cv = CitationVerificationResult(
        used_citation_ids=(99,),
        available_citation_ids=(1,),
        valid_citation_ids=(),
        invalid_citation_ids=(99,),
        unused_citation_ids=(1,),
        duplicate_citation_ids=(),
        citation_validity_rate=0.0,
        has_citations=True,
        has_valid_citations=False,
        has_invalid_citations=True,
    )
    ans = _grounded_answer(
        mode="partial",
        answer="Bad [99].",
        citations=(_citation(1, "body", None),),
        used=(),
        cv=cv,
    )
    item = evaluate_answer_question(g, ans, retrieval_mode="dense_only")
    assert not item.passed
    assert not item.mode_matches


def test_evaluate_insufficient_source_metric_none() -> None:
    g = AnswerGoldenQuestion(
        id="t",
        question="?",
        expected_mode="insufficient_context",
        expected_terms=(),
        expected_source_paths=(),
        should_be_insufficient_context=True,
    )
    ans = GroundedAnswer(
        question="q",
        answer="I do not have enough information in the provided context "
        "to answer this question.",
        mode="insufficient_context",
        citations=(),
        used_citation_ids=(),
        retrieval_mode="sparse_only",
        insufficient_context=True,
        citation_verification=empty_verification(),
    )
    item = evaluate_answer_question(g, ans, retrieval_mode="sparse_only")
    assert item.retrieved_expected_source is None
    assert item.passed


def test_summarize_answer_evaluation() -> None:
    gold_terms = (True, False, False)
    items = (
        AnswerEvaluationItem(
            question_id="a",
            question="?",
            mode="hybrid",
            answer_mode="grounded",
            expected_mode="grounded",
            mode_matches=True,
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
            passed=True,
            error=None,
        ),
        AnswerEvaluationItem(
            question_id="b",
            question="?",
            mode="hybrid",
            answer_mode="grounded",
            expected_mode=None,
            mode_matches=True,
            contains_expected_terms=True,
            missing_expected_terms=(),
            citation_validity_rate=0.5,
            has_valid_citations=True,
            has_invalid_citations=True,
            insufficient_context_matches=True,
            retrieved_expected_source=True,
            cited_source_paths=(),
            used_citation_ids=(1,),
            invalid_citation_ids=(99,),
            passed=False,
            error=None,
        ),
        AnswerEvaluationItem(
            question_id="c",
            question="?",
            mode="hybrid",
            answer_mode="error",
            expected_mode=None,
            mode_matches=False,
            contains_expected_terms=False,
            missing_expected_terms=(),
            citation_validity_rate=0.0,
            has_valid_citations=False,
            has_invalid_citations=False,
            insufficient_context_matches=False,
            retrieved_expected_source=None,
            cited_source_paths=(),
            used_citation_ids=(),
            invalid_citation_ids=(),
            passed=False,
            error="boom",
        ),
    )
    total, ans, err, pr, ma, ea, cra, isa, rsa = summarize_answer_evaluation(
        items,
        golden_has_terms=gold_terms,
    )
    assert total == 3 and ans == 2 and err == 1
    assert 0 < pr < 1
    assert rsa == 1.0
