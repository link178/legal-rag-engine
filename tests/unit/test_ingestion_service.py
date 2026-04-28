"""IngestionService end-to-end (no DB)."""

from __future__ import annotations

import pytest
from app.ingestion.errors import DocumentNotFoundError, UnsupportedDocumentTypeError
from app.ingestion.services import default_ingestion_service


def test_ingest_file_returns_document(tmp_path) -> None:
    p = tmp_path / "a.txt"
    p.write_bytes(b"hello\r\n")
    svc = default_ingestion_service()
    doc = svc.ingest_file(p)
    assert doc.source_type == "text"
    assert doc.normalized_text == "hello"
    assert doc.raw_text == "hello\r\n"
    assert len(doc.checksum) == 64
    assert "file_checksum_sha256" in doc.metadata


def test_ingest_rejects_unsupported_extension(tmp_path) -> None:
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF")
    svc = default_ingestion_service()
    with pytest.raises(UnsupportedDocumentTypeError):
        svc.ingest_file(p)


def test_ingest_missing_file(tmp_path) -> None:
    svc = default_ingestion_service()
    with pytest.raises(DocumentNotFoundError):
        svc.ingest_file(tmp_path / "nope.txt")


def test_ingest_not_a_file(tmp_path) -> None:
    d = tmp_path / "dir"
    d.mkdir()
    svc = default_ingestion_service()
    with pytest.raises(DocumentNotFoundError):
        svc.ingest_file(d)
