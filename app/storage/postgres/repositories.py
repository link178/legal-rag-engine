"""Repositories: session-injected CRUD; map domain models to ORM rows."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.models import Chunk, Document, ProcessingRun
from app.storage.postgres.models import (
    ChunkRecord,
    DocumentRecord,
    IndexManifestChunkRecord,
    IndexManifestRecord,
    ProcessingRunRecord,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class DocumentRepository:
    """Persistence for ``Document`` domain entities."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, doc: Document) -> Document:
        now = _utc_now()
        row = DocumentRecord(
            id=doc.id or uuid.uuid4(),
            source_path=doc.source_path,
            source_type=doc.source_type,
            title=doc.title,
            raw_text=doc.raw_text,
            normalized_text=doc.normalized_text,
            metadata_json=dict(doc.metadata),
            checksum=doc.checksum,
            created_by_run_id=doc.created_by_run_id,
            created_at=doc.created_at or now,
            updated_at=doc.updated_at or now,
        )
        self._session.add(row)
        self._session.flush()
        return self._to_domain(row)

    def get_by_id(self, document_id: UUID) -> Document | None:
        row = self._session.get(DocumentRecord, document_id)
        return self._to_domain(row) if row else None

    def get_by_checksum(self, checksum: str) -> Document | None:
        stmt = select(DocumentRecord).where(DocumentRecord.checksum == checksum).limit(1)
        row = self._session.scalars(stmt).first()
        return self._to_domain(row) if row else None

    def get_by_source_path_and_checksum(
        self, source_path: str, checksum: str
    ) -> Document | None:
        stmt = (
            select(DocumentRecord)
            .where(
                DocumentRecord.source_path == source_path,
                DocumentRecord.checksum == checksum,
            )
            .limit(1)
        )
        row = self._session.scalars(stmt).first()
        return self._to_domain(row) if row else None

    def list_recent(self, limit: int = 100, offset: int = 0) -> list[Document]:
        stmt = (
            select(DocumentRecord).order_by(DocumentRecord.created_at.desc()).limit(limit).offset(offset)
        )
        rows = self._session.scalars(stmt).all()
        return [self._to_domain(r) for r in rows]

    @staticmethod
    def _to_domain(row: DocumentRecord) -> Document:
        return Document(
            id=row.id,
            source_path=row.source_path,
            source_type=row.source_type,
            title=row.title,
            raw_text=row.raw_text,
            normalized_text=row.normalized_text,
            metadata=dict(row.metadata_json),
            checksum=row.checksum,
            created_by_run_id=row.created_by_run_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class ChunkRepository:
    """Persistence for ``Chunk`` domain entities."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, chunk: Chunk) -> Chunk:
        now = _utc_now()
        row = ChunkRecord(
            id=chunk.id or uuid.uuid4(),
            created_by_run_id=chunk.created_by_run_id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            chunk_text=chunk.text,
            heading=chunk.heading,
            page_number=chunk.page_number,
            chunking_strategy=chunk.chunking_strategy,
            char_count=chunk.char_count,
            token_estimate=chunk.token_estimate,
            metadata_json=dict(chunk.metadata),
            checksum=chunk.checksum,
            created_at=chunk.created_at or now,
            updated_at=chunk.updated_at or now,
        )
        self._session.add(row)
        self._session.flush()
        return self._to_domain(row)

    def list_by_document(self, document_id: UUID) -> list[Chunk]:
        stmt = (
            select(ChunkRecord)
            .where(ChunkRecord.document_id == document_id)
            .order_by(ChunkRecord.chunk_index.asc())
        )
        rows = self._session.scalars(stmt).all()
        return [self._to_domain(r) for r in rows]

    def list_by_document_and_strategy(
        self, document_id: UUID, strategy: str
    ) -> list[Chunk]:
        stmt = (
            select(ChunkRecord)
            .where(
                ChunkRecord.document_id == document_id,
                ChunkRecord.chunking_strategy == strategy,
            )
            .order_by(ChunkRecord.chunk_index.asc())
        )
        rows = self._session.scalars(stmt).all()
        return [self._to_domain(r) for r in rows]

    def count_by_document_and_strategy(self, document_id: UUID, strategy: str) -> int:
        stmt = select(func.count(ChunkRecord.id)).where(
            ChunkRecord.document_id == document_id,
            ChunkRecord.chunking_strategy == strategy,
        )
        return int(self._session.scalar(stmt) or 0)

    def list_recent(self, limit: int = 100, offset: int = 0) -> list[Chunk]:
        stmt = (
            select(ChunkRecord).order_by(ChunkRecord.created_at.desc()).limit(limit).offset(offset)
        )
        rows = self._session.scalars(stmt).all()
        return [self._to_domain(r) for r in rows]

    def list_by_chunking_strategy(
        self,
        chunking_strategy: str | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Chunk]:
        """List chunks optionally filtered by strategy (for corpus-wide indexing)."""
        stmt = select(ChunkRecord).order_by(
            ChunkRecord.document_id.asc(),
            ChunkRecord.chunk_index.asc(),
            ChunkRecord.chunking_strategy.asc(),
        )
        if chunking_strategy is not None:
            stmt = stmt.where(ChunkRecord.chunking_strategy == chunking_strategy)
        if limit is not None:
            stmt = stmt.limit(limit).offset(offset)
        rows = self._session.scalars(stmt).all()
        return [self._to_domain(r) for r in rows]

    @staticmethod
    def _to_domain(row: ChunkRecord) -> Chunk:
        return Chunk(
            id=row.id,
            created_by_run_id=row.created_by_run_id,
            document_id=row.document_id,
            chunk_index=row.chunk_index,
            text=row.chunk_text,
            chunking_strategy=row.chunking_strategy,
            char_count=row.char_count,
            heading=row.heading,
            page_number=row.page_number,
            token_estimate=row.token_estimate,
            metadata=dict(row.metadata_json),
            checksum=row.checksum,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class ProcessingRunRepository:
    """Persistence for ``ProcessingRun`` domain entities."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, run: ProcessingRun) -> ProcessingRun:
        now = _utc_now()
        row = ProcessingRunRecord(
            id=run.id or uuid.uuid4(),
            run_type=run.run_type,
            status=run.status,
            started_at=run.started_at or now,
            finished_at=run.finished_at,
            documents_processed=run.documents_processed,
            chunks_created=run.chunks_created,
            metadata_json=dict(run.metadata),
            error_message=run.error_message,
            created_at=run.created_at or now,
            updated_at=run.updated_at or now,
        )
        self._session.add(row)
        self._session.flush()
        return self._to_domain(row)

    def get_by_id(self, run_id: UUID) -> ProcessingRun | None:
        row = self._session.get(ProcessingRunRecord, run_id)
        return self._to_domain(row) if row else None

    def list_recent(self, limit: int = 100, offset: int = 0) -> list[ProcessingRun]:
        stmt = (
            select(ProcessingRunRecord)
            .order_by(ProcessingRunRecord.started_at.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = self._session.scalars(stmt).all()
        return [self._to_domain(r) for r in rows]

    def mark_completed(
        self,
        run_id: UUID,
        *,
        documents_processed: int = 0,
        chunks_created: int = 0,
        metadata_patch: dict | None = None,
    ) -> ProcessingRun | None:
        row = self._session.get(ProcessingRunRecord, run_id)
        if row is None:
            return None
        now = _utc_now()
        row.status = "completed"
        row.finished_at = now
        row.updated_at = now
        row.documents_processed = documents_processed
        row.chunks_created = chunks_created
        if metadata_patch:
            merged = dict(row.metadata_json)
            merged.update(metadata_patch)
            row.metadata_json = merged
        self._session.flush()
        return self._to_domain(row)

    def mark_failed(
        self,
        run_id: UUID,
        error_message: str,
        *,
        documents_processed: int = 0,
        chunks_created: int = 0,
        metadata_patch: dict | None = None,
    ) -> ProcessingRun | None:
        row = self._session.get(ProcessingRunRecord, run_id)
        if row is None:
            return None
        now = _utc_now()
        row.status = "failed"
        row.finished_at = now
        row.updated_at = now
        row.error_message = error_message
        row.documents_processed = documents_processed
        row.chunks_created = chunks_created
        if metadata_patch:
            merged = dict(row.metadata_json)
            merged.update(metadata_patch)
            row.metadata_json = merged
        self._session.flush()
        return self._to_domain(row)

    @staticmethod
    def _to_domain(row: ProcessingRunRecord) -> ProcessingRun:
        return ProcessingRun(
            id=row.id,
            run_type=row.run_type,
            status=row.status,
            started_at=row.started_at,
            finished_at=row.finished_at,
            documents_processed=row.documents_processed,
            chunks_created=row.chunks_created,
            metadata=dict(row.metadata_json),
            error_message=row.error_message,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class IndexManifestRepository:
    """Persistence for indexing manifests (Phase 4A trace tables)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_latest_by_manifest_hash(self, mh: str) -> IndexManifestRecord | None:
        stmt = (
            select(IndexManifestRecord)
            .where(IndexManifestRecord.manifest_hash == mh)
            .order_by(IndexManifestRecord.created_at.desc())
            .limit(1)
        )
        return self._session.scalars(stmt).first()

    def add_manifest_row(
        self,
        *,
        run_id: UUID | None,
        corpus_version: str | None,
        chunking_strategy: str | None,
        embedding_provider: str,
        embedding_dimensions: int,
        document_count: int,
        chunk_count: int,
        indexed_chunk_count: int,
        include_sparse: bool,
        include_dense: bool,
        config_hash: str,
        chunk_set_hash: str,
        manifest_hash: str,
        metadata_json: dict,
    ) -> IndexManifestRecord:
        now = _utc_now()
        row = IndexManifestRecord(
            id=uuid.uuid4(),
            run_id=run_id,
            corpus_version=corpus_version,
            chunking_strategy=chunking_strategy,
            embedding_provider=embedding_provider,
            embedding_dimensions=embedding_dimensions,
            document_count=document_count,
            chunk_count=chunk_count,
            indexed_chunk_count=indexed_chunk_count,
            include_sparse=include_sparse,
            include_dense=include_dense,
            config_hash=config_hash,
            chunk_set_hash=chunk_set_hash,
            manifest_hash=manifest_hash,
            metadata_json=dict(metadata_json),
            created_at=now,
            updated_at=now,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def add_manifest_chunk_row(
        self,
        *,
        manifest_id: UUID,
        chunk_id: UUID,
        dense_indexed: bool,
        sparse_indexed: bool,
        sparse_terms: dict[str, int] | None,
        error_message: str | None = None,
    ) -> None:
        now = _utc_now()
        self._session.add(
            IndexManifestChunkRecord(
                id=uuid.uuid4(),
                manifest_id=manifest_id,
                chunk_id=chunk_id,
                dense_indexed=dense_indexed,
                sparse_indexed=sparse_indexed,
                sparse_terms_json=dict(sparse_terms) if sparse_terms is not None else None,
                error_message=error_message,
                created_at=now,
                updated_at=now,
            )
        )
