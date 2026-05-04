"""Build engine RetrievalConfig from API params + Settings (CLI parity)."""

from __future__ import annotations

from app.api.schemas.retrieve import RetrievalParams
from app.core.config import Settings
from app.retrieval.models import RetrievalConfig


def retrieval_config_from_params(params: RetrievalParams, settings: Settings) -> RetrievalConfig:
    """Mirror ``app.retrieval.cli`` / ``app.generation.cli`` manifest filter defaults."""
    raw_model = settings.embedding_model
    model_norm = (raw_model or "").strip() or None

    return RetrievalConfig(
        mode=params.mode,
        top_k=params.top_k,
        dense_top_k=params.dense_top_k,
        sparse_top_k=params.sparse_top_k,
        rrf_k=params.rrf_k,
        index_manifest_id=params.index_manifest_id,
        embedding_provider=settings.embedding_provider,
        embedding_model=model_norm,
        embedding_dimensions=settings.embedding_dimensions,
        chunking_strategy=params.chunking_strategy,
    )
