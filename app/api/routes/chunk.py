"""POST /v1/chunk — thin wrapper over chunking runner."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.errors import ChunkConfigConflictError, ChunkingFailedError
from app.api.mappers import chunking_run_result_to_response
from app.api.schemas.chunk import ChunkRequest, ChunkResponse
from app.chunking.models import ChunkingConfig
from app.chunking.runner import chunk_document_persisted
from app.core.config import Settings, get_settings
from app.domain.models import Chunk
from app.ingestion.errors import DocumentNotFoundError as IngestDocumentNotFoundError
from app.storage.postgres.repositories import ChunkRepository
from app.storage.postgres.session import session_scope

router = APIRouter()


@router.post("/chunk", response_model=ChunkResponse)
def chunk_document(
    body: ChunkRequest,
    settings: Settings = Depends(get_settings),
) -> ChunkResponse:
    cfg = ChunkingConfig(
        strategy=body.strategy,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
        min_chunk_chars=body.min_chunk_chars,
        preserve_headings=body.preserve_headings,
    )
    cfg_hash = cfg.config_hash()
    result = chunk_document_persisted(
        body.document_id, cfg, database_url=settings.database_url
    )

    if result.error == "Document not found":
        raise IngestDocumentNotFoundError(
            f"No persisted document with id {body.document_id}"
        )
    if result.error and "chunking_config_hash mismatch" in result.error:
        raise ChunkConfigConflictError(result.error)
    if result.error:
        raise ChunkingFailedError(result.error)

    total_chunks = 0
    persisted: list[Chunk] = []
    with session_scope(settings.database_url) as session:
        repo = ChunkRepository(session)
        total_chunks = repo.count_by_document_and_strategy(
            body.document_id, result.strategy
        )
        if body.include_chunks:
            persisted = repo.list_by_document_and_strategy(
                body.document_id, result.strategy
            )

    return chunking_run_result_to_response(
        result,
        document_id=body.document_id,
        config_hash=cfg_hash,
        total_chunks=total_chunks,
        persisted_chunks=persisted if body.include_chunks else None,
        include_chunks=body.include_chunks,
        preview_chars=body.chunk_preview_chars,
    )
