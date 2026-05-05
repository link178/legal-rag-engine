"""Dense retrieval over ``chunk_embeddings`` (pgvector L2 distance)."""

from __future__ import annotations

import math

from sqlalchemy.orm import Session

from app.indexing.dense.factory import build_embedding_provider
from app.indexing.models import IndexingConfig
from app.retrieval.errors import EmbeddingDimensionMismatchError, EmptyQueryError
from app.retrieval.models import RetrievalConfig, RetrievedChunk
from app.storage.postgres.models import ChunkRecord, DocumentRecord, IndexManifestRecord
from app.storage.postgres.repositories import ChunkEmbeddingRepository


def _to_retrieved_dense(
    chunk: ChunkRecord,
    doc: DocumentRecord,
    distance: float,
    rank: int,
) -> RetrievedChunk:
    score = 1.0 / (1.0 + distance)
    return RetrievedChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.chunk_text,
        source_path=doc.source_path,
        title=doc.title,
        heading=chunk.heading,
        chunk_index=chunk.chunk_index,
        chunking_strategy=chunk.chunking_strategy,
        dense_score=score,
        sparse_score=None,
        rrf_score=None,
        rank_position=rank,
        retrieval_sources=("dense",),
        metadata={"dense_distance": distance},
    )


class DenseRetriever:
    """Top-k dense hits for one resolved manifest."""

    def __init__(self, session: Session) -> None:
        self._embed_repo = ChunkEmbeddingRepository(session)

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

        ix_cfg = IndexingConfig(
            embedding_provider=manifest.embedding_provider,
            embedding_dimensions=manifest.embedding_dimensions,
            embedding_model=manifest.embedding_model,
            include_dense=True,
            include_sparse=False,
        )
        provider = build_embedding_provider(ix_cfg)
        qv = provider.embed_query(query)
        if len(qv) != manifest.embedding_dimensions:
            raise EmbeddingDimensionMismatchError(
                f"query embedding dim {len(qv)} != manifest {manifest.embedding_dimensions}"
            )

        rows = self._embed_repo.find_similar(
            index_manifest_id=mid,
            query_vector=qv,
            limit=config.dense_top_k,
            metadata_filter=config.metadata_filter,
        )
        out: list[RetrievedChunk] = []
        for i, (chunk, doc, dist) in enumerate(rows, start=1):
            d = float(dist)
            if not math.isfinite(d):
                continue
            out.append(_to_retrieved_dense(chunk, doc, d, i))
        return out
