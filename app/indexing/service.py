"""Pure indexing over domain chunks: embeddings and sparse term maps in memory."""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.models import Chunk
from app.indexing.dense.base import EmbeddingProvider
from app.indexing.dense.mock_provider import DeterministicHashEmbeddingProvider
from app.indexing.models import IndexedChunkResult, IndexingConfig
from app.indexing.sparse.document_terms import extract_sparse_terms


class IndexingService:
    """Compute dense embeddings and/or sparse term counts without database access."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embedding = embedding_provider

    def index_chunks(
        self,
        chunks: Sequence[Chunk],
        config: IndexingConfig,
    ) -> tuple[list[IndexedChunkResult], list[list[float]] | None, list[dict[str, int]] | None]:
        """
        Process chunks according to ``config``.

        Returns (results, dense_vectors_optional, sparse_maps_optional).

        When ``include_dense`` is True, dense_vectors has one row per chunk (same order).
        When ``include_sparse`` is True, sparse_maps has one dict per chunk (same order).
        """
        dense_out: list[list[float]] | None = [] if config.include_dense else None
        sparse_out: list[dict[str, int]] | None = [] if config.include_sparse else None

        if config.include_dense:
            texts = [c.text for c in chunks]
            vectors = self._embedding.embed_texts(texts)
            assert dense_out is not None
            if len(vectors) != len(chunks):
                raise ValueError("embedding provider returned wrong number of vectors")
            dense_out.extend(vectors)

        results: list[IndexedChunkResult] = []
        for c in chunks:
            if c.id is None:
                raise ValueError("chunk id is required for indexing")
            if config.include_sparse:
                assert sparse_out is not None
                sparse_out.append(extract_sparse_terms(c.text))
            results.append(
                IndexedChunkResult(
                    chunk_id=c.id,
                    dense_indexed=config.include_dense,
                    sparse_indexed=config.include_sparse,
                    error=None,
                )
            )

        return results, dense_out, sparse_out


def default_indexing_service(dimensions: int = 16) -> IndexingService:
    return IndexingService(DeterministicHashEmbeddingProvider(dimensions=dimensions))

