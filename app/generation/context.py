"""Build citable context blocks from retrieval hits (pure; no DB / LLM)."""

from __future__ import annotations

from collections.abc import Sequence

from app.generation.models import GroundedContextBlock
from app.retrieval.models import RetrievedChunk


def _branch_score(hit: RetrievedChunk) -> float | None:
    """Comparable score: rrf > dense > sparse (mirrors evaluation reporting)."""
    if hit.rrf_score is not None:
        return hit.rrf_score
    if hit.dense_score is not None:
        return hit.dense_score
    if hit.sparse_score is not None:
        return hit.sparse_score
    return None


class ContextBuilder:
    """Convert ranked ``RetrievedChunk`` rows into citation blocks for prompting."""

    def __init__(
        self,
        max_chunks: int = 5,
        max_context_chars: int = 8000,
        max_chunk_chars: int = 2000,
        min_score: float | None = None,
    ) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be > 0")
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be > 0")
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be > 0")
        self._max_chunks = max_chunks
        self._max_context_chars = max_context_chars
        self._max_chunk_chars = max_chunk_chars
        self._min_score = min_score

    def build(self, retrieved_chunks: Sequence[RetrievedChunk]) -> list[GroundedContextBlock]:
        blocks: list[GroundedContextBlock] = []
        used_chars = 0
        citation_id = 1

        for hit in retrieved_chunks:
            if len(blocks) >= self._max_chunks:
                break

            score = _branch_score(hit)
            if self._min_score is not None:
                if score is None:
                    continue
                if score < self._min_score:
                    continue

            raw = hit.text
            truncated = raw
            if len(raw) > self._max_chunk_chars:
                truncated = raw[: self._max_chunk_chars - 1] + "\u2026"

            if len(truncated) + used_chars > self._max_context_chars:
                break

            blocks.append(
                GroundedContextBlock(
                    citation_id=citation_id,
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    source_path=hit.source_path,
                    title=hit.title,
                    heading=hit.heading,
                    rank=hit.rank_position,
                    score=score,
                    text=truncated,
                )
            )
            used_chars += len(truncated)
            citation_id += 1

        return blocks
