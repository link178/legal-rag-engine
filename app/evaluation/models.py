"""Retrieval evaluation domain types (frozen dataclasses; no Pydantic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


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
    hit_at_k_notes: str = field(
        default="hit_rate counts questions with hit_rank <= top_k among answered_questions"
    )
