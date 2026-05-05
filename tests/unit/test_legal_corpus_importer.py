"""Legal corpus importer (unit tests with fakes)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from app.domain.models import Document, ProcessingRun
from app.ingestion.adapters.legal_corpus.importer import LegalCorpusImporter
from app.ingestion.runner import IngestionRunResult
from app.ingestion.services import default_ingestion_service


def test_importer_loaded_happy(tmp_path) -> None:
    root = tmp_path / "corp"
    root.mkdir()
    (root / "a.txt").write_text("hello\n", encoding="utf-8")

    svc = default_ingestion_service()
    summary = LegalCorpusImporter(svc).import_corpus(root, persist=False)

    assert summary.discovered_count == 1
    assert summary.failed_count == 0
    assert summary.imported_count == 0
    assert summary.reused_count == 0
    assert len(summary.items) == 1
    assert summary.items[0].status == "loaded"
    assert summary.items[0].metadata.get("corpus_adapter") == "legalize"


def test_importer_limit(tmp_path) -> None:
    root = tmp_path / "corp"
    root.mkdir()
    for i in range(5):
        (root / f"f{i}.txt").write_text(f"x{i}", encoding="utf-8")

    svc = default_ingestion_service()
    summary = LegalCorpusImporter(svc, limit=2).import_corpus(root, persist=False)

    assert summary.discovered_count == 5
    assert len(summary.items) == 2


def test_importer_fail_fast_loaded(tmp_path) -> None:
    root = tmp_path / "corp"
    root.mkdir()
    (root / "good.txt").write_text("ok\n", encoding="utf-8")
    (root / "bad.txt").write_bytes(b"\xff\xfe")

    svc = default_ingestion_service()
    summary = LegalCorpusImporter(svc, fail_fast=True).import_corpus(root, persist=False)

    assert summary.failed_count >= 1
    assert len(summary.items) == 1


class _FakeRecorder:
    def __init__(self) -> None:
        self.umbrella_id = uuid4()
        self.completed: list[tuple[UUID, object]] = []
        self.failed: list[tuple[UUID, str, object | None]] = []

    def start(self, corpus_path: Path, corpus_name: str) -> ProcessingRun:
        return ProcessingRun(
            run_type="corpus_import",
            status="running",
            started_at=datetime.now(UTC),
            id=self.umbrella_id,
        )

    def mark_completed(self, umbrella_id: UUID, summary: object) -> None:
        self.completed.append((umbrella_id, summary))

    def mark_failed(self, umbrella_id: UUID, msg: str, summary: object | None = None) -> None:
        self.failed.append((umbrella_id, msg, summary))


def test_importer_persist_forwards_extra_metadata_and_corpus_run(tmp_path) -> None:
    root = tmp_path / "corp"
    root.mkdir()
    (root / "a.txt").write_text("x\n", encoding="utf-8")
    (root / "b.txt").write_text("y\n", encoding="utf-8")

    calls: list[tuple[str, dict | None, UUID | None]] = []

    rec = _FakeRecorder()

    def fake_ingest(
        path: Path,
        service,
        *,
        database_url=None,
        extra_metadata=None,
        corpus_run_id=None,
    ) -> IngestionRunResult:
        calls.append((path.name, dict(extra_metadata) if extra_metadata else None, corpus_run_id))
        doc = Document(
            source_path=str(path.resolve()),
            source_type="text",
            checksum="b" * 64 if path.name == "b.txt" else "a" * 64,
            id=uuid4(),
            title=path.stem,
            raw_text="raw",
            normalized_text="norm",
            metadata=dict(extra_metadata or {}),
        )
        return IngestionRunResult(document=doc, run=None, created=True, skipped=False, error=None)

    svc = default_ingestion_service()
    importer = LegalCorpusImporter(
        svc,
        ingest_callable=fake_ingest,
        run_recorder=rec,
        database_url=None,
    )
    summary = importer.import_corpus(root, persist=True)

    assert summary.imported_count == 2
    assert summary.corpus_run_id == str(rec.umbrella_id)
    assert len(calls) == 2
    assert calls[0][2] == rec.umbrella_id
    assert calls[0][1] is not None and calls[0][1]["corpus_adapter"] == "legalize"
    assert len(rec.completed) == 1
    assert rec.failed == []


def test_importer_fail_fast_persist_marks_failed(tmp_path) -> None:
    root = tmp_path / "corp"
    root.mkdir()
    (root / "a.txt").write_text("x\n", encoding="utf-8")
    (root / "b.txt").write_text("y\n", encoding="utf-8")

    rec = _FakeRecorder()
    state = {"n": 0}

    def fake_ingest(path, service, *, database_url=None, extra_metadata=None, corpus_run_id=None):
        state["n"] += 1
        if state["n"] == 2:
            return IngestionRunResult(
                document=None,
                run=None,
                created=False,
                skipped=False,
                error="boom",
            )
        doc = Document(
            source_path=str(path.resolve()),
            source_type="text",
            checksum="a" * 64,
            id=uuid4(),
            title="t",
            raw_text="r",
            normalized_text="n",
            metadata={},
        )
        return IngestionRunResult(document=doc, run=None, created=True, skipped=False, error=None)

    svc = default_ingestion_service()
    importer = LegalCorpusImporter(
        svc,
        ingest_callable=fake_ingest,
        run_recorder=rec,
        fail_fast=True,
    )
    summary = importer.import_corpus(root, persist=True)

    assert summary.failed_count == 1
    assert len(summary.items) == 2
    assert rec.failed
    assert rec.completed == []
