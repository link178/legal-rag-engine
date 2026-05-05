"""Pure answer evaluation metrics (no DB / I/O)."""

from __future__ import annotations

from collections.abc import Sequence

from app.evaluation.models import AnswerEvaluationItem, AnswerGoldenQuestion
from app.generation.models import GroundedAnswer, empty_verification


def _normalize_whitespace_lower(s: str) -> str:
    return " ".join(s.lower().split())


def terms_match_contains(
    answer_text: str,
    citation_previews: Sequence[str],
    terms: tuple[str, ...],
    *,
    expected_terms_in: str,
) -> tuple[bool, tuple[str, ...]]:
    """
    Whether every non-empty term appears where ``expected_terms_in`` allows.

    For ``both``, a term matches if present in normalized answer OR in any normalized
    citation preview.
    """
    non_empty = [t.strip() for t in terms if t.strip()]
    if not non_empty:
        return True, ()
    missing: list[str] = []
    for t in non_empty:
        tn = _normalize_whitespace_lower(t)
        haystacks: list[str] = []
        if expected_terms_in in ("answer", "both"):
            haystacks.append(_normalize_whitespace_lower(answer_text))
        if expected_terms_in in ("citations", "both"):
            for p in citation_previews:
                haystacks.append(_normalize_whitespace_lower(p))
        if not haystacks or not any(tn in h for h in haystacks):
            missing.append(t)
    return (not missing), tuple(missing)


def _source_paths_expectation_met(
    paths: Sequence[str], cited_paths: Sequence[str | None]
) -> bool:
    """Expected path strip must match as ci substring of some citation ``source_path``."""
    non_empty_expect = [p.strip() for p in paths if p.strip()]
    if not non_empty_expect:
        return True
    lowered_cited = [((sp or "").lower()) for sp in cited_paths]
    for pt in non_empty_expect:
        pt_low = pt.lower()
        if not any(pt_low in c for c in lowered_cited):
            return False
    return True


def compute_pass(
    *,
    error: str | None,
    answer_mode: str,
    expected_mode: str | None,
    mode_matches: bool,
    insufficient_context_matches: bool,
    contains_expected_terms: bool,
    expected_terms: tuple[str, ...],
    has_valid_citations: bool,
    has_invalid_citations: bool,
    retrieved_expected_source: bool | None,
    expected_source_paths: tuple[str, ...],
) -> bool:
    """
    Composite pass gate (explicit; keep in sync with Phase 11 docs).

    When ``expected_source_paths`` is non-empty, ``retrieved_expected_source`` must be
    exactly ``True`` (``None`` from insufficient-context N/A counts as failure for that gate).
    """
    if error is not None:
        return False
    if expected_mode is not None and not mode_matches:
        return False
    if not insufficient_context_matches:
        return False
    if expected_terms and not contains_expected_terms:
        return False
    if answer_mode == "grounded" and (
        not has_valid_citations or has_invalid_citations
    ):
        return False
    if expected_source_paths:
        if retrieved_expected_source is not True:
            return False
    return True


def evaluate_answer_question(
    golden: AnswerGoldenQuestion,
    answer: GroundedAnswer,
    *,
    retrieval_mode: str,
) -> AnswerEvaluationItem:
    """Grade one golden vs one ``GroundedAnswer``."""
    previews = tuple(c.text_preview for c in answer.citations)
    contains_terms, missing = terms_match_contains(
        answer.answer,
        previews,
        golden.expected_terms,
        expected_terms_in=golden.expected_terms_in,
    )
    cv = answer.citation_verification
    if cv is None:
        cv = empty_verification()
    cited_paths = tuple(c.source_path for c in answer.citations)

    answered_mode_matches = golden.expected_mode is None or answer.mode == golden.expected_mode
    insufficient_ok = answer.insufficient_context == golden.should_be_insufficient_context

    retrieved: bool | None
    if answer.mode == "insufficient_context":
        retrieved = None
    elif golden.expected_source_paths:
        retrieved = _source_paths_expectation_met(golden.expected_source_paths, cited_paths)
    else:
        retrieved = True

    has_expected_terms_nonempty = bool(
        golden.expected_terms and any(t.strip() for t in golden.expected_terms)
    )
    contains_for_item = contains_terms if has_expected_terms_nonempty else True

    passed = compute_pass(
        error=None,
        answer_mode=answer.mode,
        expected_mode=golden.expected_mode,
        mode_matches=answered_mode_matches,
        insufficient_context_matches=insufficient_ok,
        contains_expected_terms=contains_for_item,
        expected_terms=tuple(p for p in golden.expected_terms if p.strip()),
        has_valid_citations=cv.has_valid_citations,
        has_invalid_citations=cv.has_invalid_citations,
        retrieved_expected_source=retrieved,
        expected_source_paths=golden.expected_source_paths,
    )

    return AnswerEvaluationItem(
        question_id=golden.id,
        question=golden.question,
        mode=retrieval_mode,
        answer_mode=answer.mode,
        expected_mode=golden.expected_mode,
        mode_matches=answered_mode_matches,
        contains_expected_terms=contains_for_item,
        missing_expected_terms=missing,
        citation_validity_rate=cv.citation_validity_rate,
        has_valid_citations=cv.has_valid_citations,
        has_invalid_citations=cv.has_invalid_citations,
        insufficient_context_matches=insufficient_ok,
        retrieved_expected_source=retrieved,
        cited_source_paths=cited_paths,
        used_citation_ids=answer.used_citation_ids,
        invalid_citation_ids=tuple(cv.invalid_citation_ids),
        passed=passed,
        error=None,
    )


def summarize_answer_evaluation(
    items: Sequence[AnswerEvaluationItem],
    *,
    golden_has_terms: Sequence[bool],
) -> tuple[
    int,
    int,
    int,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """
    Aggregate metrics over items ordered like ``golden_has_terms``.

    ``golden_has_terms[k]``: original golden row had non-empty ``expected_terms``.
    """
    total = len(items)
    errors = sum(1 for i in items if i.error is not None)
    answered = total - errors
    if answered == 0:
        return total, 0, errors, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    answered_items = tuple(i for i in items if i.error is None)
    passes = sum(1 for i in answered_items if i.passed)
    pass_rate = passes / answered

    mode_accuracy = sum(1 for i in answered_items if i.mode_matches) / answered

    ins_acc = (
        sum(1 for i in answered_items if i.insufficient_context_matches) / answered
    )

    cit_avg = sum(i.citation_validity_rate for i in answered_items) / answered

    term_idxs = tuple(
        k for k in range(total) if k < len(golden_has_terms) and golden_has_terms[k]
    )
    term_answered = tuple(k for k in term_idxs if items[k].error is None)
    if term_answered:
        expected_terms_accuracy = (
            sum(1 for k in term_answered if items[k].contains_expected_terms)
            / len(term_answered)
        )
    else:
        expected_terms_accuracy = 1.0

    src_idxs = tuple(
        k
        for k in range(total)
        if items[k].error is None and items[k].retrieved_expected_source is not None
    )
    if src_idxs:
        retrieved_expected_source_rate = (
            sum(
                1
                for k in src_idxs
                if items[k].retrieved_expected_source is True
            )
            / len(src_idxs)
        )
    else:
        retrieved_expected_source_rate = 1.0

    return (
        total,
        answered,
        errors,
        pass_rate,
        mode_accuracy,
        expected_terms_accuracy,
        cit_avg,
        ins_acc,
        retrieved_expected_source_rate,
    )
