"""Operator-triggered ingest + optional Postgres persistence (one file per run)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.domain.models import Document, ProcessingRun
from app.ingestion.errors import IngestionError
from app.ingestion.services import IngestionService
from app.storage.postgres.repositories import DocumentRepository, ProcessingRunRepository
from app.storage.postgres.session import session_scope


@dataclass(slots=True)
class IngestionRunResult:
    document: Document | None
    run: ProcessingRun | None
    created: bool
    skipped: bool
    error: str | None = None


def ingest_file_persisted(
    path: Path,
    ingestion_service: IngestionService,
    *,
    database_url: str | None = None,
) -> IngestionRunResult:
    """Ingest one file inside a DB transaction; record ``ProcessingRun`` and optional ``Document``.

    Idempotent on ``(source_path, checksum)``: existing row is returned with ``skipped=True``.
    New rows set ``Document.created_by_run_id`` to the run that created them.
    """
    now = datetime.now(UTC)
    run_domain = ProcessingRun(
        run_type="ingest",
        status="running",
        started_at=now,
        metadata={"source": "ingestion.runner"},
    )
    try:
        with session_scope(database_url) as session:
            run_repo = ProcessingRunRepository(session)
            doc_repo = DocumentRepository(session)
            persisted_run = run_repo.add(run_domain)
            run_id = persisted_run.id
            assert run_id is not None

            try:
                doc = ingestion_service.ingest_file(path)
            except IngestionError as e:
                run_repo.mark_failed(
                    run_id,
                    str(e),
                    metadata_patch={"path": str(path.resolve())},
                )
                return IngestionRunResult(
                    document=None,
                    run=run_repo.get_by_id(run_id),
                    created=False,
                    skipped=False,
                    error=str(e),
                )

            existing = doc_repo.get_by_source_path_and_checksum(doc.source_path, doc.checksum)
            if existing is not None:
                run_repo.mark_completed(
                    run_id,
                    documents_processed=0,
                    chunks_created=0,
                    metadata_patch={"skipped": True, "document_id": str(existing.id)},
                )
                return IngestionRunResult(
                    document=existing,
                    run=run_repo.get_by_id(run_id),
                    created=False,
                    skipped=True,
                    error=None,
                )

            to_save = Document(
                source_path=doc.source_path,
                source_type=doc.source_type,
                checksum=doc.checksum,
                title=doc.title,
                raw_text=doc.raw_text,
                normalized_text=doc.normalized_text,
                metadata=doc.metadata,
                created_by_run_id=run_id,
            )
            saved = doc_repo.add(to_save)
            run_repo.mark_completed(
                run_id,
                documents_processed=1,
                chunks_created=0,
                metadata_patch={"document_id": str(saved.id)},
            )
            return IngestionRunResult(
                document=saved,
                run=run_repo.get_by_id(run_id),
                created=True,
                skipped=False,
                error=None,
            )
    except Exception as e:  # noqa: BLE001 — DB connectivity etc.; transaction may roll back
        return IngestionRunResult(
            document=None,
            run=None,
            created=False,
            skipped=False,
            error=f"{type(e).__name__}: {e}",
        )
