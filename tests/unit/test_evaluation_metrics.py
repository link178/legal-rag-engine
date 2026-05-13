"""Pure retrieval evaluation metrics."""

from __future__ import annotations

from uuid import uuid4

from app.evaluation.metrics.retrieval import (
    compute_hit_at_k,
    compute_reciprocal_rank,
    evaluate_question,
    summarize_retrieval_evaluation,
)
from app.evaluation.models import RetrievalEvaluationItem, RetrievalGoldenQuestion
from app.retrieval.models import RetrievedChunk


def _hit(
    *,
    chunk_id,
    document_id,
    text: str = "hello",
    source_path: str | None = "p.md",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        source_path=source_path,
        title=None,
        heading=None,
        chunk_index=0,
        chunking_strategy="fixed_size",
        dense_score=0.5,
        rank_position=1,
        retrieval_sources=("dense",),
    )


def test_hit_by_chunk_id() -> None:
    cid = uuid4()
    did = uuid4()
    g = RetrievalGoldenQuestion(id="q", question="q?", expected_chunk_ids=(cid,))
    results = [
        RetrievedChunk(
            chunk_id=uuid4(),
            document_id=did,
            text="nope",
            source_path=None,
            title=None,
            heading=None,
            chunk_index=0,
            chunking_strategy="fixed_size",
            dense_score=0.1,
            rank_position=1,
            retrieval_sources=("dense",),
        ),
        _hit(chunk_id=cid, document_id=did, text="x"),
    ]
    hit, rank, matched = evaluate_question(g, results, top_k=5)
    assert hit and rank == 2 and "chunk_id" in matched


def test_hit_by_document_id() -> None:
    cid = uuid4()
    did = uuid4()
    g = RetrievalGoldenQuestion(id="q", question="q?", expected_document_ids=(did,))
    results = [_hit(chunk_id=cid, document_id=did)]
    hit, rank, matched = evaluate_question(g, results, top_k=5)
    assert hit and rank == 1 and "document_id" in matched


def test_hit_by_source_path_substring() -> None:
    cid = uuid4()
    did = uuid4()
    g = RetrievalGoldenQuestion(
        id="q",
        question="q?",
        expected_source_paths=("intro.md",),
    )
    results = [_hit(chunk_id=cid, document_id=did, source_path="data/basic/intro.md")]
    hit, rank, matched = evaluate_question(g, results, top_k=5)
    assert hit and rank == 1 and "source_path" in matched


def test_hit_by_expected_terms_all_in_one_chunk() -> None:
    cid = uuid4()
    did = uuid4()
    g = RetrievalGoldenQuestion(
        id="q",
        question="q?",
        expected_terms=("foo", "bar"),
    )
    results = [_hit(chunk_id=cid, document_id=did, text="This has foo and bar inside.")]
    hit, rank, matched = evaluate_question(g, results, top_k=5)
    assert hit and rank == 1 and "term" in matched


def test_miss_when_no_match() -> None:
    cid = uuid4()
    did = uuid4()
    g = RetrievalGoldenQuestion(id="q", question="q?", expected_terms=("nope",))
    results = [_hit(chunk_id=cid, document_id=did, text="other")]
    hit, rank, matched = evaluate_question(g, results, top_k=5)
    assert not hit and rank is None and matched == ()


def test_rr_rank_1_and_2_and_miss() -> None:
    assert compute_reciprocal_rank(1) == 1.0
    assert compute_reciprocal_rank(2) == 0.5
    assert compute_reciprocal_rank(None) == 0.0


def test_hit_at_k() -> None:
    assert compute_hit_at_k(1, k=5) is True
    assert compute_hit_at_k(6, k=5) is False


def test_summarize_hit_rate_and_mrr() -> None:
    items = (
        RetrievalEvaluationItem(
            question_id="a",
            question="?",
            mode="hybrid",
            top_k=5,
            hit=True,
            hit_rank=1,
            reciprocal_rank=1.0,
            matched_by=("term",),
            retrieved_chunk_ids=(),
            retrieved_document_ids=(),
            retrieved_source_paths=(),
            retrieved_scores=(),
            error=None,
        ),
        RetrievalEvaluationItem(
            question_id="b",
            question="?",
            mode="hybrid",
            top_k=5,
            hit=False,
            hit_rank=None,
            reciprocal_rank=0.0,
            matched_by=(),
            retrieved_chunk_ids=(),
            retrieved_document_ids=(),
            retrieved_source_paths=(),
            retrieved_scores=(),
            error=None,
        ),
        RetrievalEvaluationItem(
            question_id="c",
            question="?",
            mode="hybrid",
            top_k=5,
            hit=False,
            hit_rank=None,
            reciprocal_rank=0.0,
            matched_by=(),
            retrieved_chunk_ids=(),
            retrieved_document_ids=(),
            retrieved_source_paths=(),
            retrieved_scores=(),
            error="boom",
        ),
    )
    total, answered, errored, hit_rate, mrr = summarize_retrieval_evaluation(items)
    assert total == 3
    assert answered == 2
    assert errored == 1
    assert hit_rate == 0.5
    assert mrr == 0.5
