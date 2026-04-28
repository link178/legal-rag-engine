"""File metadata helpers."""

from __future__ import annotations

from app.ingestion.parsers.metadata import extract_basic_file_metadata, infer_source_type


def test_infer_source_type(tmp_path) -> None:
    assert infer_source_type(tmp_path / "a.txt") == "text"
    assert infer_source_type(tmp_path / "b.md") == "markdown"
    assert infer_source_type(tmp_path / "c.markdown") == "markdown"
    assert infer_source_type(tmp_path / "d.pdf") == "unknown"


def test_extract_basic_file_metadata(tmp_path) -> None:
    p = tmp_path / "doc.txt"
    p.write_bytes(b"abcd")
    meta = extract_basic_file_metadata(p)
    assert meta["filename"] == "doc.txt"
    assert meta["extension"] == "txt"
    assert meta["file_size_bytes"] == 4
