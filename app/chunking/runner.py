"""Operator-triggered chunk persistence: one ``ProcessingRun`` per chunk job."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from app.chunking.errors import ChunkingError
from app.chunking.models import ChunkingConfig
from app.chunking.service import default_chunking_service
from app.domain.models import Chunk, ProcessingRun
from app.storage.postgres.repositories import (
    ChunkRepository,
    DocumentRepository,
    ProcessingRunRepository,
)
from app.storage.postgres.session import session_scope


@dataclass(slots=True)
class ChunkingRunResult:
    """Outcome of ``chunk_document_persisted``."""

    document_id: UUID | None
    run: ProcessingRun | None
    strategy: str
    created_chunks: int
    skipped_existing: bool
    chunk_ids: list[UUID]
    error: str | None = None


def chunk_document_persisted(
    document_id: UUID,
    config: ChunkingConfig,
    *,
    database_url: str | None = None,
) -> ChunkingRunResult:
    """Load a persisted document by id, chunk it, persist chunks under a ``ProcessingRun``.

    Idempotent when existing chunks for ``(document_id, strategy)`` share the same
    ``chunking_config_hash`` in metadata. Different config hashes are rejected without
    deleting existing rows.
    """
    strategy_name = config.strategy.strip()
    now = datetime.now(UTC)
    run_domain = ProcessingRun(
        run_type="chunking",
        status="running",
        started_at=now,
        metadata={
            "source": "chunking.runner",
            "document_id": str(document_id),
            "strategy": strategy_name,
            "chunking_config_hash": config.config_hash(),
        },
    )
    svc = default_chunking_service()

    try:
        with session_scope(database_url) as session:
            run_repo = ProcessingRunRepository(session)
            doc_repo = DocumentRepository(session)
            chunk_repo = ChunkRepository(session)

            persisted_run = run_repo.add(run_domain)
            run_id = persisted_run.id
            assert run_id is not None

            doc = doc_repo.get_by_id(document_id)
            if doc is None:
                run_repo.mark_failed(
                    run_id,
                    "Document not found",
                    metadata_patch={"document_id": str(document_id)},
                )
                return ChunkingRunResult(
                    document_id=document_id,
                    run=run_repo.get_by_id(run_id),
                    strategy=strategy_name,
                    created_chunks=0,
                    skipped_existing=False,
                    chunk_ids=[],
                    error="Document not found",
                )

            incoming_hash = config.config_hash()
            existing = chunk_repo.list_by_document_and_strategy(document_id, strategy_name)

            if existing:
                stored_hash = existing[0].metadata.get("chunking_config_hash")
                if stored_hash == incoming_hash:
                    run_repo.mark_completed(
                        run_id,
                        documents_processed=1,
                        chunks_created=0,
                        metadata_patch={
                            "skipped": True,
                            "document_id": str(document_id),
                            "chunk_count_existing": len(existing),
                        },
                    )
                    return ChunkingRunResult(
                        document_id=document_id,
                        run=run_repo.get_by_id(run_id),
                        strategy=strategy_name,
                        created_chunks=0,
                        skipped_existing=True,
                        chunk_ids=[c.id for c in existing if c.id],
                        error=None,
                    )
                msg = (
                    "Existing chunks for this document and strategy were produced with a "
                    "different chunking configuration (chunking_config_hash mismatch)."
                )
                run_repo.mark_failed(
                    run_id,
                    msg,
                    metadata_patch={"document_id": str(document_id)},
                )
                return ChunkingRunResult(
                    document_id=document_id,
                    run=run_repo.get_by_id(run_id),
                    strategy=strategy_name,
                    created_chunks=0,
                    skipped_existing=False,
                    chunk_ids=[],
                    error=msg,
                )

            try:
                chunks = svc.chunk_document(doc, config)
            except ChunkingError as e:
                run_repo.mark_failed(
                    run_id,
                    str(e),
                    metadata_patch={"document_id": str(document_id)},
                )
                return ChunkingRunResult(
                    document_id=document_id,
                    run=run_repo.get_by_id(run_id),
                    strategy=strategy_name,
                    created_chunks=0,
                    skipped_existing=False,
                    chunk_ids=[],
                    error=str(e),
                )

            saved_ids: list[UUID] = []
            for ch in chunks:
                to_save = Chunk(
                    document_id=ch.document_id,
                    chunk_index=ch.chunk_index,
                    text=ch.text,
                    chunking_strategy=ch.chunking_strategy,
                    char_count=ch.char_count,
                    heading=ch.heading,
                    page_number=ch.page_number,
                    token_estimate=ch.token_estimate,
                    metadata=dict(ch.metadata),
                    checksum=ch.checksum,
                    created_by_run_id=run_id,
                )
                saved = chunk_repo.add(to_save)
                assert saved.id is not None
                saved_ids.append(saved.id)

            run_repo.mark_completed(
                run_id,
                documents_processed=1,
                chunks_created=len(saved_ids),
                metadata_patch={
                    "document_id": str(document_id),
                    "chunk_ids": [str(x) for x in saved_ids],
                },
            )
            return ChunkingRunResult(
                document_id=document_id,
                run=run_repo.get_by_id(run_id),
                strategy=strategy_name,
                created_chunks=len(saved_ids),
                skipped_existing=False,
                chunk_ids=saved_ids,
                error=None,
            )
    except Exception as e:  # noqa: BLE001 — DB connectivity etc.; transaction may roll back
        return ChunkingRunResult(
            document_id=document_id,
            run=None,
            strategy=strategy_name,
            created_chunks=0,
            skipped_existing=False,
            chunk_ids=[],
            error=f"{type(e).__name__}: {e}",
        )
