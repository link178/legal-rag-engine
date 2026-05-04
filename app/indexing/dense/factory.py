"""Build a dense ``EmbeddingProvider`` from ``IndexingConfig`` (branch-local imports)."""

from __future__ import annotations

from app.indexing.dense.base import EmbeddingProvider
from app.indexing.dense.mock_provider import DeterministicHashEmbeddingProvider
from app.indexing.models import IndexingConfig


def build_embedding_provider(config: IndexingConfig) -> EmbeddingProvider:
    """
    Return an instance implementing ``EmbeddingProvider`` for the given config.

    Does not import optional ``sentence_transformers`` unless the provider requires it.
    """
    provider_id = config.embedding_provider.strip()
    if provider_id == "deterministic_hash":
        return DeterministicHashEmbeddingProvider(dimensions=config.embedding_dimensions)
    if provider_id == "local_sentence_transformers":
        from app.indexing.dense.local_provider import SentenceTransformersEmbeddingProvider

        model = config.normalized_embedding_model()
        if not model:
            raise ValueError(
                "embedding_model is required when embedding_provider is "
                "'local_sentence_transformers'"
            )
        return SentenceTransformersEmbeddingProvider(
            model_name=model,
            dimensions=config.embedding_dimensions,
        )
    raise ValueError(f"Unsupported embedding_provider: {provider_id!r}")
