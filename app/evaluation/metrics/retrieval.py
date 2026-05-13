"""Pure retrieval evaluation metrics (no DB / I/O)."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from app.evaluation.models import RetrievalEvaluationItem, RetrievalGoldenQuestion
from app.retrieval.models import RetrievedChunk


def _normalize_whitespace_lower(s: str) -> str:
    return " ".join(s.lower().split())


def _terms_all_in_chunk(text: str, terms: tuple[str, ...]) -> bool:
    """True if every non-empty expected term is a case-insensitive substring of normalized text."""
    norm = _normalize_whitespace_lower(text)
    non_empty = [t.strip() for t in terms if t.strip()]
    if not non_empty:
        return False
    for t in non_empty:
        tn = _normalize_whitespace_lower(t)
        if tn not in norm:
            return False
    return True


def _matched_criteria(hit: RetrievedChunk, golden: RetrievalGoldenQuestion) -> list[str]:
    out: list[str] = []
    if golden.expected_chunk_ids and hit.chunk_id in golden.expected_chunk_ids:
        out.append("chunk_id")
    if golden.expected_document_ids and hit.document_id in golden.expected_document_ids:
        out.append("document_id")
    sp = (hit.source_path or "").lower()
    for p in golden.expected_source_paths:
        pt = p.strip()
        if pt and pt.lower() in sp:
            out.append("source_path")
            break
    if golden.expected_terms and _terms_all_in_chunk(hit.text, golden.expected_terms):
        out.append("term")
    # Dedupe preserving first-seen order
    seen: set[str] = set()
    deduped: list[str] = []
    for m in out:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


def evaluate_question(
    golden: RetrievalGoldenQuestion,
    results: Sequence[RetrievedChunk],
    *,
    top_k: int,
) -> tuple[bool, int | None, tuple[str, ...]]:
    """
    Grade golden question against ranked hits (1-based positions in ``results[:top_k]``).

    Returns (hit, hit_rank, matched_by). First matching chunk wins; ``matched_by`` lists all
    criteria that matched that chunk.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    consider = list(results)[:top_k]
    for pos, hit in enumerate(consider, start=1):
        crit = _matched_criteria(hit, golden)
        if crit:
            return True, pos, tuple(crit)
    return False, None, ()


def compute_hit_at_k(hit_rank: int | None, k: int) -> bool:
    """Hit@k: relevant chunk appears in position <= k (1-based)."""
    if k <= 0:
        raise ValueError("k must be > 0")
    return hit_rank is not None and 1 <= hit_rank <= k


def compute_reciprocal_rank(hit_rank: int | None) -> float:
    """MRR contribution for one question: 1/rank if hit, else 0."""
    if hit_rank is None or hit_rank < 1:
        return 0.0
    return 1.0 / float(hit_rank)


def summarize_retrieval_evaluation(
    items: Sequence[RetrievalEvaluationItem],
) -> tuple[int, int, int, float, float]:
    """
    Return (total_questions, answered_questions, errored_questions, hit_rate, mrr).

    ``hit_rate`` = answered-hits / answered (questions with error excluded from denominator).
    ``mrr`` = mean reciprocal rank over answered only (errors contribute 0 RR via items).
    """
    total = len(items)
    errored = sum(1 for i in items if i.error is not None)
    answered = total - errored
    if answered == 0:
        return total, 0, errored, 0.0, 0.0
    hits = sum(1 for i in items if i.error is None and i.hit)
    hit_rate = hits / answered
    mrr = sum(i.reciprocal_rank for i in items if i.error is None) / answered
    return total, answered, errored, hit_rate, mrr


def branch_score(hit: RetrievedChunk) -> float | None:
    """Comparable score for reporting: rrf > dense > sparse."""
    if hit.rrf_score is not None:
        return hit.rrf_score
    if hit.dense_score is not None:
        return hit.dense_score
    if hit.sparse_score is not None:
        return hit.sparse_score
    return None


def extracted_lists_from_hits(
    hits: Sequence[RetrievedChunk],
) -> tuple[tuple[UUID, ...], tuple[UUID, ...], tuple[str | None, ...], tuple[float | None, ...]]:
    """Build parallel tuples for RetrievalEvaluationItem from retrieved chunks."""
    cids: list[UUID] = []
    dids: list[UUID] = []
    paths: list[str | None] = []
    scores: list[float | None] = []
    for h in hits:
        cids.append(h.chunk_id)
        dids.append(h.document_id)
        paths.append(h.source_path)
        scores.append(branch_score(h))
    return tuple(cids), tuple(dids), tuple(paths), tuple(scores)
