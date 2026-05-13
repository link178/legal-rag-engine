"""Frontmatter parsing (minimal scalar YAML-like block)."""

from __future__ import annotations

from app.ingestion.adapters.legal_corpus.metadata import parse_frontmatter


def test_parse_frontmatter_basic() -> None:
    raw = "---\njurisdiction: eu\nanswer: true\nyear: 2024\n---\n\n# Hi\n"
    data, warns = parse_frontmatter(raw)
    assert warns == []
    assert data["jurisdiction"] == "eu"
    assert data["answer"] is True
    assert data["year"] == 2024


def test_parse_frontmatter_quotes() -> None:
    raw = "---\ntitle: \"Quoted\"\n---\n\nbody\n"
    data, warns = parse_frontmatter(raw)
    assert data["title"] == "Quoted"


def test_parse_frontmatter_no_block() -> None:
    raw = "# Just markdown\n\n---\nnot: frontmatter\n"
    data, warns = parse_frontmatter(raw)
    assert data == {}
    assert warns == []


def test_parse_frontmatter_unclosed() -> None:
    raw = "---\nx: y\nno closing\n"
    data, warns = parse_frontmatter(raw)
    assert data == {}
    assert any("unclosed" in w for w in warns)


def test_parse_frontmatter_skips_bad_lines_with_warning() -> None:
    raw = "---\nvalid: ok\nnot valid line\n---\n\n# X\n"
    data, warns = parse_frontmatter(raw)
    assert data["valid"] == "ok"
    assert warns
