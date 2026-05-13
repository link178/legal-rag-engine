"""RetrievalGoldenQuestion and summary model validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from app.evaluation.models import (
    RetrievalEvaluationItem,
    RetrievalEvaluationSummary,
    RetrievalGoldenQuestion,
)


def test_golden_question_valid_terms() -> None:
    g = RetrievalGoldenQuestion(
        id="q1",
        question="What is X?",
        expected_terms=("a", "b"),
    )
    assert g.id == "q1"
    assert g.expected_terms == ("a", "b")


def test_golden_question_rejects_no_expectations() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RetrievalGoldenQuestion(id="q1", question="What?")


def test_golden_question_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="id"):
        RetrievalGoldenQuestion(id="  ", question="x", expected_terms=("t",))


def test_golden_question_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="question text"):
        RetrievalGoldenQuestion(id="q", question=" \n ", expected_terms=("t",))


def test_summary_json_roundtrip_shape() -> None:
    item = RetrievalEvaluationItem(
        question_id="q1",
        question="q?",
        mode="hybrid",
        top_k=5,
        hit=True,
        hit_rank=1,
        reciprocal_rank=1.0,
        matched_by=("term",),
        retrieved_chunk_ids=(uuid4(),),
        retrieved_document_ids=(uuid4(),),
        retrieved_source_paths=("p.md",),
        retrieved_scores=(0.5,),
        error=None,
    )
    s = RetrievalEvaluationSummary(
        schema_version="evaluation.retrieval.v1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        config={"mode": "hybrid", "top_k": 5},
        manifest={"id": "00000000-0000-0000-0000-000000000001"},
        total_questions=1,
        answered_questions=1,
        errored_questions=0,
        hit_rate=1.0,
        mrr=1.0,
        items=(item,),
    )
    payload = {
        "schema_version": s.schema_version,
        "total": s.total_questions,
        "items": [
            {
                "question_id": item.question_id,
                "hit": item.hit,
                "chunk_ids": [str(x) for x in item.retrieved_chunk_ids],
            }
        ],
    }
    dumped = json.dumps(payload)
    loaded = json.loads(dumped)
    assert loaded["schema_version"] == "evaluation.retrieval.v1"
    assert loaded["items"][0]["hit"] is True
