"""GET /v1/documents — list and optional detail."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies.db import get_session
from app.api.errors import PersistedDocumentNotFoundError
from app.api.mappers import chunk_record_to_response
from app.api.schemas.chunk import DocumentChunksResponse
from app.api.schemas.documents import DocumentDetailResponse, DocumentListItem, DocumentListResponse
from app.storage.postgres.models import ChunkRecord
from app.storage.postgres.repositories import ChunkRepository, DocumentRepository

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> DocumentListResponse:
    repo = DocumentRepository(session)
    docs = repo.list_recent(limit=limit, offset=offset)
    items = [
        DocumentListItem(
            id=d.id,  # type: ignore[arg-type]
            source_path=d.source_path,
            title=d.title,
            checksum=d.checksum,
            metadata=dict(d.metadata),
            created_at=d.created_at,
        )
        for d in docs
        if d.id is not None
    ]
    return DocumentListResponse(documents=items, count=len(items), limit=limit, offset=offset)


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
def get_document(
    document_id: UUID,
    session: Session = Depends(get_session),
) -> DocumentDetailResponse:
    repo = DocumentRepository(session)
    d = repo.get_by_id(document_id)
    if d is None or d.id is None:
        raise PersistedDocumentNotFoundError()

    stmt = (
        select(func.count())
        .select_from(ChunkRecord)
        .where(ChunkRecord.document_id == document_id)
    )
    chunks_count = int(session.scalar(stmt) or 0)

    return DocumentDetailResponse(
        id=d.id,
        source_path=d.source_path,
        title=d.title,
        checksum=d.checksum,
        metadata=dict(d.metadata),
        created_at=d.created_at,
        chunks_count=chunks_count,
    )


_CHUNK_LIST_PREVIEW = 240


@router.get("/documents/{document_id}/chunks", response_model=DocumentChunksResponse)
def list_document_chunks(
    document_id: UUID,
    strategy: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> DocumentChunksResponse:
    doc_repo = DocumentRepository(session)
    d = doc_repo.get_by_id(document_id)
    if d is None or d.id is None:
        raise PersistedDocumentNotFoundError()

    strategy_norm: str | None = None
    if strategy is not None:
        s = strategy.strip()
        strategy_norm = s or None

    chunk_repo = ChunkRepository(session)
    if strategy_norm:
        total = chunk_repo.count_by_document_and_strategy(document_id, strategy_norm)
        all_chunks = chunk_repo.list_by_document_and_strategy(document_id, strategy_norm)
    else:
        cnt_stmt = (
            select(func.count())
            .select_from(ChunkRecord)
            .where(ChunkRecord.document_id == document_id)
        )
        total = int(session.scalar(cnt_stmt) or 0)
        all_chunks = chunk_repo.list_by_document(document_id)

    page = all_chunks[offset : offset + limit]
    items = [
        chunk_record_to_response(ch, include_text=True, preview_chars=_CHUNK_LIST_PREVIEW)
        for ch in page
    ]

    return DocumentChunksResponse(
        document_id=document_id,
        strategy=strategy_norm,
        count=total,
        limit=limit,
        offset=offset,
        chunks=items,
    )
