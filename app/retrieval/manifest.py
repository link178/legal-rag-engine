"""Resolve ``IndexManifestRecord`` for retrieval (read path)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.retrieval.errors import ManifestNotFoundError
from app.retrieval.models import RetrievalConfig
from app.storage.postgres.models import IndexManifestRecord
from app.storage.postgres.repositories import IndexManifestRepository


def resolve_manifest_record(
    session: Session,
    config: RetrievalConfig,
    *,
    for_dense: bool,
) -> IndexManifestRecord:
    """
    Pick the manifest for this query.

    ``for_dense``: when True, require ``embeddings_persisted`` (explicit id or auto
    ``get_latest_completed``). When False (sparse-only path), use latest manifest
    with ``include_sparse`` plus optional family filters.
    """
    repo = IndexManifestRepository(session)
    if config.index_manifest_id is not None:
        row = repo.get_by_id(config.index_manifest_id)
        if row is None:
            raise ManifestNotFoundError(
                f"No index manifest with id={config.index_manifest_id}"
            )
        if for_dense and not row.embeddings_persisted:
            raise ManifestNotFoundError(
                "Index manifest has no persisted embeddings (embeddings_persisted=false). "
                "Run: python -m app.indexing.cli --chunking-strategy <name>"
            )
        return row

    if for_dense:
        row = repo.get_latest_completed(
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            embedding_dimensions=config.embedding_dimensions,
            chunking_strategy=config.chunking_strategy,
        )
    else:
        row = repo.get_latest_with_sparse(
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            embedding_dimensions=config.embedding_dimensions,
            chunking_strategy=config.chunking_strategy,
        )
    if row is None:
        raise ManifestNotFoundError(
            "No suitable index manifest found. Ingest, chunk, then run: "
            "python -m app.indexing.cli"
        )
    return row
