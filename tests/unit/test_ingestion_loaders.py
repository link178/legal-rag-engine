"""Loaders for .txt and Markdown."""

from __future__ import annotations

import pytest
from app.ingestion.errors import DocumentLoadError
from app.ingestion.loaders.markdown_loader import MarkdownLoader
from app.ingestion.loaders.text_loader import TextLoader


def test_text_loader_reads_txt(tmp_path) -> None:
    p = tmp_path / "note.txt"
    p.write_bytes(b"hello\n")
    loaded = TextLoader().load(p)
    assert loaded.source_type == "text"
    assert loaded.raw_text == "hello\n"
    assert loaded.title == "note"


def test_markdown_loader_heading_title(tmp_path) -> None:
    p = tmp_path / "x.md"
    p.write_text("# My Title\n\nbody\n", encoding="utf-8")
    loaded = MarkdownLoader().load(p)
    assert loaded.source_type == "markdown"
    assert loaded.title == "My Title"
    assert "body" in loaded.raw_text


def test_markdown_loader_fallback_title_from_filename(tmp_path) -> None:
    p = tmp_path / "no_heading.md"
    p.write_text("just text\n", encoding="utf-8")
    loaded = MarkdownLoader().load(p)
    assert loaded.title == "no_heading"


def test_loader_invalid_utf8(tmp_path) -> None:
    p = tmp_path / "bad.txt"
    p.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(DocumentLoadError):
        TextLoader().load(p)


def test_default_ingestion_service_resolves_new_extensions(tmp_path) -> None:
    from pathlib import Path

    from app.ingestion.services import default_ingestion_service

    fixture_pdf = Path(__file__).resolve().parent.parent / "fixtures" / "sample.pdf"

    svc = default_ingestion_service()
    h = tmp_path / "x.html"
    h.write_text("<html><body><p>h</p></body></html>", encoding="utf-8")
    assert svc.ingest_file(h).source_type == "html"

    hm = tmp_path / "y.htm"
    hm.write_text("<html><body><p>j</p></body></html>", encoding="utf-8")
    assert svc.ingest_file(hm).source_type == "html"

    assert svc.ingest_file(fixture_pdf).source_type == "pdf"


def test_default_ingestion_service_rejects_docx(tmp_path) -> None:
    from app.ingestion.errors import UnsupportedDocumentTypeError
    from app.ingestion.services import default_ingestion_service

    p = tmp_path / "nope.docx"
    p.write_bytes(b"x")
    with pytest.raises(UnsupportedDocumentTypeError):
        default_ingestion_service().ingest_file(p)
