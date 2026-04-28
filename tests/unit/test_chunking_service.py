"""Unit tests for ``ChunkingService`` routing (no DB)."""

from __future__ import annotations

import uuid

import pytest
from app.chunking.errors import UnsupportedChunkingStrategyError
from app.chunking.models import ChunkingConfig
from app.chunking.service import ChunkingService, default_chunking_service
from app.chunking.strategies.fixed_size import FixedSizeChunkingStrategy
from app.domain.models import Document


def _doc(text: str = "hello") -> Document:
    return Document(
        id=uuid.uuid4(),
        source_path="/tmp/x.md",
        source_type="markdown",
        checksum="abc",
        normalized_text=text,
    )


def test_select_fixed_size() -> None:
    svc = default_chunking_service()
    cfg = ChunkingConfig(strategy="fixed_size")
    chunks = svc.chunk_document(_doc(), cfg)
    assert chunks and chunks[0].chunking_strategy == "fixed_size"


def test_select_structure_aware() -> None:
    svc = default_chunking_service()
    cfg = ChunkingConfig(strategy="structure_aware")
    chunks = svc.chunk_document(_doc("# H\n\nbody"), cfg)
    assert chunks


def test_rejects_unknown_strategy() -> None:
    svc = ChunkingService({"fixed_size": FixedSizeChunkingStrategy()})
    cfg = ChunkingConfig(strategy="does_not_exist")
    with pytest.raises(UnsupportedChunkingStrategyError):
        svc.chunk_document(_doc(), cfg)


def test_structure_aware_registered_in_default_service() -> None:
    svc = default_chunking_service()
    cfg = ChunkingConfig(strategy="structure_aware")
    chunks = svc.chunk_document(
        Document(
            id=uuid.uuid4(),
            source_path="/tmp/x.md",
            source_type="markdown",
            checksum="abc",
            normalized_text="# H\n\nbody text\n",
        ),
        cfg,
    )
    assert chunks and chunks[0].chunking_strategy == "structure_aware"
