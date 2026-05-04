"""POST /v1/ingest — thin wrapper over ingestion service + runner."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from app.api.errors import IngestionFailedError
from app.api.schemas.ingest import IngestRequest, IngestResponse
from app.core.config import Settings, get_settings
from app.ingestion.errors import (
    DocumentLoadError,
    UnsupportedDocumentTypeError,
)
from app.ingestion.errors import (
    DocumentNotFoundError as IngestDocumentNotFoundError,
)
from app.ingestion.runner import ingest_file_persisted
from app.ingestion.services import default_ingestion_service

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(
    body: IngestRequest,
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    path = Path(body.path).expanduser()
    if not path.exists() or not path.is_file():
        raise IngestDocumentNotFoundError(f"Path does not exist or is not a file: {body.path}")

    if not body.persist:
        try:
            doc = default_ingestion_service().ingest_file(path)
        except (UnsupportedDocumentTypeError, DocumentLoadError):
            raise
        return IngestResponse(
            persisted=False,
            document_id=None,
            processing_run_id=None,
            source_path=doc.source_path,
            source_type=doc.source_type,
            checksum=doc.checksum,
            title=doc.title,
            metadata=dict(doc.metadata),
            created=None,
            skipped=None,
            error=None,
        )

    result = ingest_file_persisted(
        path,
        default_ingestion_service(),
        database_url=settings.database_url,
    )
    if result.error:
        raise IngestionFailedError(result.error)
    if result.document is None:
        raise IngestionFailedError("ingestion returned no document")

    d = result.document
    run_id = result.run.id if result.run and result.run.id else None
    doc_id = d.id
    return IngestResponse(
        persisted=True,
        document_id=doc_id,
        processing_run_id=run_id,
        source_path=d.source_path,
        source_type=d.source_type,
        checksum=d.checksum,
        title=d.title,
        metadata=dict(d.metadata),
        created=result.created,
        skipped=result.skipped,
        error=None,
    )
