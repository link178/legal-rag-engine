"""Retrieval + answer evaluation domain types (frozen dataclasses; no Pydantic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from app.core.constants import SUPPORTED_RETRIEVAL_MODES
from app.generation.models import SUPPORTED_ANSWER_MODES

ANSWER_EXPECTED_TERMS_IN: tuple[str, ...] = ("answer", "citations", "both")


@dataclass(frozen=True, kw_only=True, slots=True)
class RetrievalGoldenQuestion:
    """One golden question for retrieval-only evaluation (JSONL row)."""

    id: str
    question: str
    expected_terms: tuple[str, ...] = ()
    expected_document_ids: tuple[UUID, ...] = ()
    expected_chunk_ids: tuple[UUID, ...] = ()
    expected_source_paths: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        qid = self.id.strip()
        if not qid:
            raise ValueError("golden question id must be non-empty")
        object.__setattr__(self, "id", qid)
        q = self.question.strip()
        if not q:
            raise ValueError("golden question text must be non-empty")
        object.__setattr__(self, "question", q)
        if not (
            self.expected_terms
            or self.expected_document_ids
            or self.expected_chunk_ids
            or self.expected_source_paths
        ):
            raise ValueError(
                "at least one of expected_terms, expected_document_ids, "
                "expected_chunk_ids, expected_source_paths must be non-empty"
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class RetrievalEvaluationItem:
    """Per-question outcome after retrieval + grading."""

    question_id: str
    question: str
    mode: str
    top_k: int
    hit: bool
    hit_rank: int | None
    reciprocal_rank: float
    matched_by: tuple[str, ...]
    retrieved_chunk_ids: tuple[UUID, ...]
    retrieved_document_ids: tuple[UUID, ...]
    retrieved_source_paths: tuple[str | None, ...]
    retrieved_scores: tuple[float | None, ...]
    error: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class RetrievalEvaluationSummary:
    """Full report: traceability + aggregates + per-question items."""

    schema_version: str
    created_at: datetime
    config: dict[str, Any]
    manifest: dict[str, Any]
    total_questions: int
    answered_questions: int
    errored_questions: int
    hit_rate: float
    mrr: float
    items: tuple[RetrievalEvaluationItem, ...]
    execution_status: str = "FAILED"
    hit_at_k_notes: str = field(
        default="hit_rate counts questions with hit_rank <= top_k among answered_questions"
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class AnswerGoldenQuestion:
    """One golden row for answer-level evaluation (JSONL)."""

    id: str
    question: str
    expected_mode: str | None = None
    expected_terms: tuple[str, ...] = ()
    expected_terms_in: str = "both"
    expected_source_paths: tuple[str, ...] = ()
    should_be_insufficient_context: bool = False
    recommended_mode: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        qid = self.id.strip()
        if not qid:
            raise ValueError("golden question id must be non-empty")
        object.__setattr__(self, "id", qid)
        q = self.question.strip()
        if not q:
            raise ValueError("golden question text must be non-empty")
        object.__setattr__(self, "question", q)
        if self.expected_mode is not None and self.expected_mode not in SUPPORTED_ANSWER_MODES:
            raise ValueError(
                f"expected_mode must be one of {SUPPORTED_ANSWER_MODES} or null; "
                f"got {self.expected_mode!r}"
            )
        eti = (self.expected_terms_in or "").strip().lower()
        if eti not in ANSWER_EXPECTED_TERMS_IN:
            raise ValueError(
                f"expected_terms_in must be one of {ANSWER_EXPECTED_TERMS_IN}; "
                f"got {self.expected_terms_in!r}"
            )
        object.__setattr__(self, "expected_terms_in", eti)
        rm = self.recommended_mode
        if rm is not None:
            rms = rm.strip().lower()
            if rms not in SUPPORTED_RETRIEVAL_MODES:
                raise ValueError(
                    f"recommended_mode must be one of {SUPPORTED_RETRIEVAL_MODES} "
                    f"or null; got {self.recommended_mode!r}"
                )
            object.__setattr__(self, "recommended_mode", rms)
        notes_norm = None if self.notes is None else str(self.notes)
        if notes_norm is not None and not notes_norm.strip():
            notes_norm = None
        object.__setattr__(self, "notes", notes_norm)


@dataclass(frozen=True, kw_only=True, slots=True)
class AnswerEvaluationItem:
    """Per-question outcome after GroundedAnswerer + grading."""

    question_id: str
    question: str
    mode: str
    answer_mode: str
    expected_mode: str | None
    mode_matches: bool
    contains_expected_terms: bool
    missing_expected_terms: tuple[str, ...]
    citation_validity_rate: float
    has_valid_citations: bool
    has_invalid_citations: bool
    insufficient_context_matches: bool
    retrieved_expected_source: bool | None
    cited_source_paths: tuple[str | None, ...]
    used_citation_ids: tuple[int, ...]
    invalid_citation_ids: tuple[int, ...]
    passed: bool
    error: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class AnswerEvaluationSummary:
    """Full answer evaluation report."""

    schema_version: str
    created_at: datetime
    config: dict[str, Any]
    manifest: dict[str, Any]
    total_questions: int
    answered_questions: int
    errored_questions: int
    pass_rate: float
    mode_accuracy: float
    expected_terms_accuracy: float
    citation_validity_rate_avg: float
    insufficient_context_accuracy: float
    retrieved_expected_source_rate: float
    items: tuple[AnswerEvaluationItem, ...]
    execution_status: str = "FAILED"
    retrieved_expected_source_notes: str = field(
        default=(
            "retrieved_expected_source_rate is over applicable items only "
            "(excludes insufficient_context where retrieved_expected_source is null)"
        )
    )
