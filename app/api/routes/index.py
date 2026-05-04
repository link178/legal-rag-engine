"""POST /v1/index and GET /v1/index-manifests — thin wrapper over indexing runner + repo."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.dependencies.db import get_session
from app.api.errors import IndexingFailedError, NoChunksToIndexError
from app.api.mappers import (
    index_manifest_record_to_detail,
    index_manifest_record_to_item,
    indexing_run_result_to_response,
)
from app.api.schemas.index import (
    IndexManifestDetailResponse,
    IndexManifestListResponse,
    IndexRequest,
    IndexResponse,
)
from app.core.config import Settings, get_settings
from app.indexing.models import IndexingConfig
from app.indexing.runner import index_chunks_persisted
from app.retrieval.errors import ManifestNotFoundError
from app.storage.postgres.repositories import IndexManifestRepository

router = APIRouter()


def _resolve_indexing_config(body: IndexRequest, settings: Settings) -> IndexingConfig:
    provider = (
        body.embedding_provider
        if body.embedding_provider is not None
        else settings.embedding_provider
    )
    dims = (
        body.embedding_dimensions
        if body.embedding_dimensions is not None
        else settings.embedding_dimensions
    )
    raw_model = (
        body.embedding_model if body.embedding_model is not None else settings.embedding_model
    )
    model_norm = (raw_model or "").strip() or None
    strat = body.chunking_strategy
    if strat is not None:
        s = strat.strip()
        strat = s or None

    return IndexingConfig(
        embedding_provider=provider,
        embedding_dimensions=dims,
        embedding_model=model_norm,
        chunking_strategy=strat,
        batch_size=body.batch_size,
        include_dense=body.include_dense,
        include_sparse=body.include_sparse,
        force_reindex=body.force_reindex,
    )


@router.post("/index", response_model=IndexResponse)
def run_index(
    body: IndexRequest,
    settings: Settings = Depends(get_settings),
) -> IndexResponse:
    cfg = _resolve_indexing_config(body, settings)
    result = index_chunks_persisted(
        cfg,
        database_url=settings.database_url,
        corpus_version=body.corpus_version,
    )

    if result.error and "No chunks match the indexing filter" in result.error:
        raise NoChunksToIndexError(result.error)
    if result.error:
        raise IndexingFailedError(result.error)

    return indexing_run_result_to_response(
        result,
        fallback_provider=settings.embedding_provider,
        fallback_dims=settings.embedding_dimensions,
        fallback_model=(settings.embedding_model or "").strip() or None,
    )


@router.get("/index-manifests", response_model=IndexManifestListResponse)
def list_index_manifests(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    chunking_strategy: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = Query(default=None),
    include_sparse: bool | None = None,
    include_dense: bool | None = None,
    session: Session = Depends(get_session),
) -> IndexManifestListResponse:
    repo = IndexManifestRepository(session)
    rows = repo.list_recent(
        limit=limit,
        offset=offset,
        chunking_strategy=chunking_strategy,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        include_sparse=include_sparse,
        include_dense=include_dense,
    )
    items = [index_manifest_record_to_item(r) for r in rows]
    return IndexManifestListResponse(
        manifests=items, count=len(items), limit=limit, offset=offset
    )


@router.get("/index-manifests/{manifest_id}", response_model=IndexManifestDetailResponse)
def get_index_manifest(
    manifest_id: UUID,
    session: Session = Depends(get_session),
) -> IndexManifestDetailResponse:
    row = IndexManifestRepository(session).get_by_id(manifest_id)
    if row is None:
        raise ManifestNotFoundError(f"manifest {manifest_id} not found")
    return index_manifest_record_to_detail(row)
