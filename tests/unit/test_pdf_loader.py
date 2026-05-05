"""PDF loader tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from app.ingestion.errors import DocumentLoadError
from app.ingestion.loaders.pdf_loader import PdfLoader
from pypdf import PdfWriter

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample.pdf"


def test_pdf_loader_extracts_text_and_page_count() -> None:
    loaded = PdfLoader().load(_FIXTURE)
    assert loaded.source_type == "pdf"
    assert loaded.metadata.get("format") == "pdf"
    assert loaded.metadata.get("page_count") == 2
    assert "Page one text" in loaded.raw_text
    assert "Page two text" in loaded.raw_text


def test_pdf_loader_corrupt_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"not a pdf")
    with pytest.raises(DocumentLoadError, match="Failed to read PDF"):
        PdfLoader().load(p)


def test_pdf_loader_blank_pages_no_text_raises(tmp_path: Path) -> None:
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    w.add_blank_page(width=612, height=792)
    buf = BytesIO()
    w.write(buf)
    p = tmp_path / "blank.pdf"
    p.write_bytes(buf.getvalue())
    with pytest.raises(DocumentLoadError, match="no extractable text"):
        PdfLoader().load(p)


def test_pdf_loader_encrypted_raises(tmp_path: Path) -> None:
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    w.encrypt("user-secret-password")
    buf = BytesIO()
    w.write(buf)
    p = tmp_path / "enc.pdf"
    p.write_bytes(buf.getvalue())
    with pytest.raises(DocumentLoadError, match="encrypt"):
        PdfLoader().load(p)
