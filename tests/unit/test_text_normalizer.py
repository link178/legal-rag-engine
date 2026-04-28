"""Unit tests for conservative text normalization."""

from __future__ import annotations

from app.ingestion.normalizers.text_normalizer import normalize_text


def test_crlf_to_lf() -> None:
    assert normalize_text("a\r\nb") == "a\nb"


def test_trailing_spaces_per_line() -> None:
    assert normalize_text("hello   \nworld\t  ") == "hello\nworld"


def test_collapse_excessive_blank_lines() -> None:
    assert normalize_text("a\n\n\n\nb") == "a\n\nb"


def test_preserves_heading_markers() -> None:
    text = "# Title\n\nBody"
    assert normalize_text(text) == "# Title\n\nBody"


def test_outer_strip() -> None:
    assert normalize_text("\n\nhello\n") == "hello"
