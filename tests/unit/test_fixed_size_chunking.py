"""Unit tests for fixed-size chunking strategy (no DB)."""

from __future__ import annotations

import uuid

import pytest
from app.chunking.errors import DocumentIdRequiredError, EmptyDocumentError
from app.chunking.models import ChunkingConfig
from app.chunking.strategies.fixed_size import FixedSizeChunkingStrategy
from app.domain.models import Document


def _doc(text: str, *, doc_id: uuid.UUID | None = None) -> Document:
    did = doc_id or uuid.uuid4()
    return Document(
        id=did,
        source_path="/tmp/x.md",
        source_type="markdown",
        checksum="abc123",
        normalized_text=text,
        raw_text=text,
    )


def test_short_document_single_chunk() -> None:
    strat = FixedSizeChunkingStrategy()
    cfg = ChunkingConfig(strategy="fixed_size", chunk_size=1200, chunk_overlap=200)
    doc = _doc("hello world")
    chunks = strat.split(doc, cfg)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "hello world"
    assert chunks[0].checksum
    assert chunks[0].metadata["start_offset"] == 0
    assert chunks[0].metadata["end_offset"] == len("hello world")
    assert chunks[0].metadata["chunking_config_hash"] == cfg.config_hash()


def test_long_document_multiple_chunks_and_overlap() -> None:
    strat = FixedSizeChunkingStrategy()
    cfg = ChunkingConfig(strategy="fixed_size", chunk_size=40, chunk_overlap=10, min_chunk_chars=5)
    body = "word " * 80  # long enough for multiple windows
    doc = _doc(body.strip())
    chunks = strat.split(doc, cfg)
    assert len(chunks) >= 2
    assert all(c.chunking_strategy == "fixed_size" for c in chunks)
    assert chunks[0].token_estimate >= 1


def test_checksum_stable() -> None:
    strat = FixedSizeChunkingStrategy()
    cfg = ChunkingConfig(strategy="fixed_size", chunk_size=400, chunk_overlap=50)
    doc = _doc("deterministic body text.\n" * 30)
    c1 = strat.split(doc, cfg)[0].checksum
    c2 = strat.split(doc, cfg)[0].checksum
    assert c1 == c2


def test_empty_text_raises() -> None:
    strat = FixedSizeChunkingStrategy()
    cfg = ChunkingConfig()
    doc = _doc("   ")
    doc = Document(
        id=doc.id,
        source_path=doc.source_path,
        source_type=doc.source_type,
        checksum=doc.checksum,
        normalized_text="   ",
        raw_text=None,
    )
    with pytest.raises(EmptyDocumentError):
        strat.split(doc, cfg)


def test_missing_document_id_raises() -> None:
    strat = FixedSizeChunkingStrategy()
    cfg = ChunkingConfig()
    doc = Document(
        id=None,
        source_path="/tmp/x.md",
        source_type="markdown",
        checksum="abc",
        normalized_text="hello",
    )
    with pytest.raises(DocumentIdRequiredError):
        strat.split(doc, cfg)
