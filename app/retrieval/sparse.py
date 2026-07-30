"""Sparse / lexical retrieval from persisted ``index_manifest_chunks.sparse_terms_json``.

When ``RetrievalConfig.metadata_filter`` is set, only chunks whose documents match
the filter participate in BM25-lite scoring; ``N`` and document frequencies are
computed over that filtered subset (IDF is consistent within the subspace).
"""

from __future__ import annotations

import math

from sqlalchemy.orm import Session

from app.indexing.sparse.tokenizer import tokenize
from app.retrieval.errors import EmptyQueryError
from app.retrieval.models import RetrievalConfig, RetrievedChunk
from app.storage.postgres.models import ChunkRecord, DocumentRecord, IndexManifestRecord
from app.storage.postgres.repositories import IndexManifestRepository


def _idf(N: int, df: int) -> float:
    return math.log((N - df + 0.5) / (df + 0.5) + 1.0)


def _score_chunk(
    query_terms: list[str],
    idf_by_term: dict[str, float],
    chunk_terms: dict[str, int],
) -> tuple[float, int]:
    """Return (score, number of distinct query terms present in chunk)."""
    seen = 0
    total = 0.0
    for t in query_terms:
        idf = idf_by_term.get(t)
        if idf is None:
            continue
        tf = chunk_terms.get(t, 0)
        if tf <= 0:
            continue
        seen += 1
        total += idf * (tf / (tf + 1.0))
    return total, seen


class SparseRetriever:
    """BM25-lite over term counts stored at indexing time."""

    def __init__(self, session: Session) -> None:
        self._manifest_repo = IndexManifestRepository(session)

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig,
        manifest: IndexManifestRecord,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise EmptyQueryError("query must be non-empty")
        mid = manifest.id
        if mid is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        rows = self._manifest_repo.list_sparse_chunk_rows(
            mid, metadata_filter=config.metadata_filter
        )
        if not rows:
            return []

        N = len(rows)
        unique_q = list(dict.fromkeys(query_tokens))
        df: dict[str, int] = {t: 0 for t in unique_q}
        for _c, _d, terms in rows:
            for t in unique_q:
                if terms.get(t, 0) > 0:
                    df[t] += 1

        idf_by_term = {t: _idf(N, df[t]) for t in unique_q}

        scored: list[tuple[ChunkRecord, DocumentRecord, float, int]] = []
        for chunk, doc, terms in rows:
            sc, nmatch = _score_chunk(query_tokens, idf_by_term, terms)
            scored.append((chunk, doc, sc, nmatch))

        scored.sort(
            key=lambda x: (
                -x[2],
                x[1].source_path or "",
                x[0].chunk_index if x[0].chunk_index is not None else -1,
                str(x[0].id) if x[0].id is not None else "",
            )
        )
        positive = [(c, d, sc, n) for c, d, sc, n in scored if sc > 0]
        out: list[RetrievedChunk] = []
        for i, (chunk, doc, sc, nmatch) in enumerate(positive[: config.sparse_top_k], start=1):
            out.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.chunk_text,
                    source_path=doc.source_path,
                    title=doc.title,
                    heading=chunk.heading,
                    chunk_index=chunk.chunk_index,
                    chunking_strategy=chunk.chunking_strategy,
                    dense_score=None,
                    sparse_score=sc,
                    rrf_score=None,
                    rank_position=i,
                    retrieval_sources=("sparse",),
                    metadata={"sparse_terms_matched": nmatch},
                )
            )
        return out
