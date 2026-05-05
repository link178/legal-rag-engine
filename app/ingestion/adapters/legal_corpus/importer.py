"""Batch import for legal corpus directories."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from app.domain.models import ProcessingRun
from app.ingestion.adapters.legal_corpus.discovery import scan_corpus
from app.ingestion.adapters.legal_corpus.metadata import (
    infer_corpus_metadata,
    load_frontmatter_for_candidate,
)
from app.ingestion.adapters.legal_corpus.models import (
    LegalCorpusDocumentCandidate,
    LegalCorpusImportItem,
    LegalCorpusImportStatus,
    LegalCorpusImportSummary,
)
from app.ingestion.errors import IngestionError
from app.ingestion.runner import IngestionRunResult, ingest_file_persisted
from app.ingestion.services import IngestionService
from app.storage.postgres.repositories import ProcessingRunRepository
from app.storage.postgres.session import session_scope


def _build_umbrella_metadata_patch(summary: LegalCorpusImportSummary) -> dict[str, Any]:
    return {
        "corpus_path": summary.corpus_path,
        "corpus_name": summary.corpus_name,
        "discovered_count": summary.discovered_count,
        "imported_count": summary.imported_count,
        "reused_count": summary.reused_count,
        "failed_count": summary.failed_count,
        "documents_processed": len(summary.items),
    }


class DefaultCorpusRunRecorder:
    """Creates one umbrella ``processing_runs`` row per corpus import."""

    def __init__(self, database_url: str | None = None) -> None:
        self._database_url = database_url

    def start(self, corpus_path: Path, corpus_name: str) -> ProcessingRun:
        now = datetime.now(UTC)
        run_domain = ProcessingRun(
            run_type="corpus_import",
            status="running",
            started_at=now,
            metadata={
                "corpus_path": str(corpus_path.resolve()),
                "corpus_name": corpus_name,
                "source": "legal_corpus.importer",
            },
        )
        with session_scope(self._database_url) as session:
            repo = ProcessingRunRepository(session)
            return repo.add(run_domain)

    def mark_completed(self, umbrella_id: UUID, summary: LegalCorpusImportSummary) -> None:
        patch = _build_umbrella_metadata_patch(summary)
        with session_scope(self._database_url) as session:
            repo = ProcessingRunRepository(session)
            repo.mark_completed(
                umbrella_id,
                documents_processed=patch["documents_processed"],
                chunks_created=0,
                metadata_patch=patch,
            )

    def mark_failed(
        self,
        umbrella_id: UUID,
        error_message: str,
        summary: LegalCorpusImportSummary | None = None,
    ) -> None:
        patch = _build_umbrella_metadata_patch(summary) if summary else {}
        with session_scope(self._database_url) as session:
            repo = ProcessingRunRepository(session)
            repo.mark_failed(umbrella_id, error_message, metadata_patch=patch)


class LegalCorpusImporter:
    """Discover and ingest supported files under a corpus root."""

    def __init__(
        self,
        service: IngestionService,
        *,
        ingest_callable: Callable[..., IngestionRunResult] = ingest_file_persisted,
        run_recorder: DefaultCorpusRunRecorder | None = None,
        database_url: str | None = None,
        fail_fast: bool = False,
        limit: int | None = None,
    ) -> None:
        self._service = service
        self._ingest = ingest_callable
        self._run_recorder = run_recorder
        self._database_url = database_url
        self._fail_fast = fail_fast
        self._limit = limit

    def import_corpus(self, corpus_path: Path, *, persist: bool) -> LegalCorpusImportSummary:
        root = corpus_path.expanduser().resolve()
        corpus_name = root.name

        all_candidates = scan_corpus(root)
        discovered_count = len(all_candidates)
        if self._limit is not None:
            candidates: tuple[LegalCorpusDocumentCandidate, ...] = all_candidates[
                : max(0, self._limit)
            ]
        else:
            candidates = all_candidates

        items: list[LegalCorpusImportItem] = []
        umbrella_id: UUID | None = None
        recorder = self._run_recorder
        if persist and candidates:
            if recorder is None:
                recorder = DefaultCorpusRunRecorder(self._database_url)
            umbrella = recorder.start(root, corpus_name)
            assert umbrella.id is not None
            umbrella_id = umbrella.id

        fail_stopped = False

        for candidate in candidates:
            fm_raw, fm_warnings = load_frontmatter_for_candidate(candidate)
            meta = infer_corpus_metadata(root, candidate, fm_raw)
            warnings = tuple(fm_warnings)

            if persist:
                assert umbrella_id is not None and recorder is not None
                result = self._ingest(
                    candidate.path,
                    self._service,
                    database_url=self._database_url,
                    extra_metadata=meta,
                    corpus_run_id=umbrella_id,
                )
                status, doc_id, title, err = self._map_persist_result(result)
                items.append(
                    LegalCorpusImportItem(
                        source_path=(
                            result.document.source_path
                            if result.document is not None
                            else str(candidate.path.resolve())
                        ),
                        relative_path=candidate.relative_path,
                        status=status,
                        document_id=doc_id,
                        title=title,
                        metadata=dict(meta),
                        error=err,
                        warnings=warnings,
                    )
                )
                if self._fail_fast and status == "failed":
                    partial = self._partial_summary(
                        root,
                        corpus_name,
                        discovered_count,
                        tuple(items),
                        str(umbrella_id) if umbrella_id else None,
                    )
                    recorder.mark_failed(umbrella_id, "fail_fast: ingestion error", partial)
                    fail_stopped = True
                    break
            else:
                item = self._ingest_loaded(candidate, meta, warnings)
                items.append(item)
                if self._fail_fast and item.status == "failed":
                    partial = self._partial_summary(
                        root,
                        corpus_name,
                        discovered_count,
                        tuple(items),
                        None,
                    )
                    fail_stopped = True
                    # Non-persist: no umbrella run to mark_failed
                    break

        summary = self._finalize_summary(
            root,
            corpus_name,
            discovered_count,
            tuple(items),
            str(umbrella_id) if umbrella_id else None,
        )

        if (
            persist
            and umbrella_id is not None
            and candidates
            and recorder is not None
            and not fail_stopped
        ):
            recorder.mark_completed(umbrella_id, summary)

        return summary

    @staticmethod
    def _map_persist_result(result: IngestionRunResult) -> tuple[
        LegalCorpusImportStatus,
        str | None,
        str | None,
        str | None,
    ]:
        if result.error:
            return "failed", None, None, result.error
        if result.skipped and result.document:
            did = result.document.id
            return (
                "reused",
                str(did) if did else None,
                result.document.title,
                None,
            )
        if result.document:
            did = result.document.id
            return (
                "imported",
                str(did) if did else None,
                result.document.title,
                None,
            )
        return "failed", None, None, result.error or "unknown persistence error"

    def _ingest_loaded(
        self,
        candidate: LegalCorpusDocumentCandidate,
        meta: dict[str, Any],
        warnings: tuple[str, ...],
    ) -> LegalCorpusImportItem:
        try:
            doc = self._service.ingest_file(candidate.path, extra_metadata=meta)
        except IngestionError as e:
            return LegalCorpusImportItem(
                source_path=str(candidate.path.resolve()),
                relative_path=candidate.relative_path,
                status="failed",
                document_id=None,
                title=None,
                metadata=dict(meta),
                error=str(e),
                warnings=warnings,
            )
        did = doc.id
        return LegalCorpusImportItem(
            source_path=doc.source_path,
            relative_path=candidate.relative_path,
            status="loaded",
            document_id=str(did) if did else None,
            title=doc.title,
            metadata=dict(doc.metadata),
            error=None,
            warnings=warnings,
        )

    @staticmethod
    def _partial_summary(
        corpus_root: Path,
        corpus_name: str,
        discovered_count: int,
        items: tuple[LegalCorpusImportItem, ...],
        corpus_run_id: str | None,
    ) -> LegalCorpusImportSummary:
        return LegalCorpusImporter._finalize_summary(
            corpus_root, corpus_name, discovered_count, items, corpus_run_id
        )

    @staticmethod
    def _finalize_summary(
        corpus_root: Path,
        corpus_name: str,
        discovered_count: int,
        items: tuple[LegalCorpusImportItem, ...],
        corpus_run_id: str | None,
    ) -> LegalCorpusImportSummary:
        imported = sum(1 for i in items if i.status == "imported")
        reused = sum(1 for i in items if i.status == "reused")
        failed = sum(1 for i in items if i.status == "failed")
        return LegalCorpusImportSummary(
            corpus_path=str(corpus_root.resolve()),
            corpus_name=corpus_name,
            discovered_count=discovered_count,
            imported_count=imported,
            reused_count=reused,
            failed_count=failed,
            items=items,
            corpus_run_id=corpus_run_id,
        )
