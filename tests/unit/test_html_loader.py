"""HTML loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.ingestion.errors import DocumentLoadError
from app.ingestion.loaders.html_loader import HtmlLoader


def test_html_loader_supports_html(tmp_path: Path) -> None:
    p = tmp_path / "a.html"
    p.write_text(
        "<!DOCTYPE html><html><head><title>T</title></head>"
        "<body><p>hello</p></body></html>",
        encoding="utf-8",
    )
    loaded = HtmlLoader().load(p)
    assert loaded.source_type == "html"
    assert loaded.metadata.get("format") == "html"
    assert "hello" in loaded.raw_text


def test_html_loader_supports_htm(tmp_path: Path) -> None:
    p = tmp_path / "a.htm"
    p.write_text(
        "<!DOCTYPE html><html><head><title>T</title></head>"
        "<body><p>from htm</p></body></html>",
        encoding="utf-8",
    )
    loaded = HtmlLoader().load(p)
    assert loaded.source_type == "html"
    assert "from htm" in loaded.raw_text


def test_html_loader_title_from_title_tag(tmp_path: Path) -> None:
    p = tmp_path / "t.html"
    p.write_text(
        "<html><head><title>From Title</title></head>"
        "<body><h1>Not this first for title field</h1><p>x</p></body></html>",
        encoding="utf-8",
    )
    loaded = HtmlLoader().load(p)
    assert loaded.title == "From Title"
    assert loaded.metadata.get("html_title") == "From Title"


def test_html_loader_fallback_title_h1(tmp_path: Path) -> None:
    p = tmp_path / "h.html"
    p.write_text(
        "<html><body><h1>Only H1</h1><p>body</p></body></html>",
        encoding="utf-8",
    )
    loaded = HtmlLoader().load(p)
    assert loaded.title == "Only H1"
    assert loaded.metadata.get("h1") == "Only H1"


def test_html_loader_fallback_title_filename(tmp_path: Path) -> None:
    p = tmp_path / "my_stem.html"
    p.write_text("<html><body><p>no headings</p></body></html>", encoding="utf-8")
    loaded = HtmlLoader().load(p)
    assert loaded.title == "my_stem"


def test_html_loader_strips_script_style_noscript(tmp_path: Path) -> None:
    p = tmp_path / "s.html"
    p.write_text(
        "<html><head><title>x</title><script>EVIL()</script>"
        "<style>.a{}</style></head><body><noscript>N</noscript>"
        "<p>keep me</p></body></html>",
        encoding="utf-8",
    )
    loaded = HtmlLoader().load(p)
    assert "EVIL" not in loaded.raw_text
    assert "N" not in loaded.raw_text
    assert "keep me" in loaded.raw_text


def test_html_loader_lists_and_paragraphs(tmp_path: Path) -> None:
    p = tmp_path / "l.html"
    p.write_text(
        "<html><head><title>t</title></head><body>"
        "<p>First para.</p><ul><li>A</li><li>B</li></ul></body></html>",
        encoding="utf-8",
    )
    loaded = HtmlLoader().load(p)
    assert "First para." in loaded.raw_text
    assert "A" in loaded.raw_text
    assert "B" in loaded.raw_text


def test_html_loader_latin1_with_meta_charset(tmp_path: Path) -> None:
    p = tmp_path / "enc.html"
    body = (
        "<html><head><meta charset='iso-8859-1'><title>t</title></head>"
        "<body><p>Caf\xe9</p></body></html>"
    )
    p.write_bytes(body.encode("iso-8859-1"))
    loaded = HtmlLoader().load(p)
    assert "Café" in loaded.raw_text


def test_html_loader_empty_body_raises(tmp_path: Path) -> None:
    p = tmp_path / "e.html"
    p.write_text("<html><head><title>x</title></head><body></body></html>", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="no extractable text"):
        HtmlLoader().load(p)


def test_html_loader_comment_not_in_text(tmp_path: Path) -> None:
    p = tmp_path / "c.html"
    p.write_text(
        "<html><head><title>t</title></head><body>"
        "<p>Hello <!-- secret --> world</p></body></html>",
        encoding="utf-8",
    )
    loaded = HtmlLoader().load(p)
    assert "secret" not in loaded.raw_text
    assert "Hello" in loaded.raw_text
    assert "world" in loaded.raw_text
