"""Unit tests for structure-aware Markdown chunking (no DB)."""

from __future__ import annotations

import uuid

from app.chunking.models import ChunkingConfig
from app.chunking.strategies.structure_aware import StructureAwareChunkingStrategy
from app.domain.models import Document


def _doc(text: str) -> Document:
    return Document(
        id=uuid.uuid4(),
        source_path="/tmp/x.md",
        source_type="markdown",
        checksum="abc",
        normalized_text=text,
        raw_text=text,
    )


def test_detects_hash_heading_and_assigns_heading() -> None:
    strat = StructureAwareChunkingStrategy()
    cfg = ChunkingConfig(strategy="structure_aware", chunk_size=200, chunk_overlap=20)
    text = "# Title\n\nSome body text here for the chunk.\n"
    chunks = strat.split(_doc(text), cfg)
    assert len(chunks) >= 1
    assert any(c.heading == "Title" for c in chunks)


def test_fallback_when_no_headings() -> None:
    strat = StructureAwareChunkingStrategy()
    cfg = ChunkingConfig(strategy="structure_aware", chunk_size=500, chunk_overlap=50)
    text = "no headings here " * 40
    chunks = strat.split(_doc(text), cfg)
    assert len(chunks) >= 1
    assert chunks[0].metadata.get("fallback") == "no_markdown_headings"


def test_long_section_subdivided() -> None:
    strat = StructureAwareChunkingStrategy()
    cfg = ChunkingConfig(strategy="structure_aware", chunk_size=120, chunk_overlap=10)
    body = ("paragraph text " * 80).strip()
    text = f"## Section\n\n{body}\n"
    chunks = strat.split(_doc(text), cfg)
    assert len(chunks) >= 2
