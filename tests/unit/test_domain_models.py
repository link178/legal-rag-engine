"""Unit tests for pure domain models (no database)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from app.core.constants import SUPPORTED_RETRIEVAL_MODES
from app.domain.models import Chunk, Document, ProcessingRun


def test_document_minimal() -> None:
    doc = Document(
        source_path="/corpus/a.md",
        source_type="markdown",
        checksum="sha256:abc",
        metadata={"lang": "es"},
    )
    assert doc.source_path == "/corpus/a.md"
    assert doc.metadata == {"lang": "es"}
    assert doc.id is None


def test_document_invalid_empty_source_path() -> None:
    with pytest.raises(ValueError, match="source_path"):
        Document(source_path="  ", source_type="md", checksum="x")


def test_chunk_minimal() -> None:
    did = uuid.uuid4()
    ch = Chunk(
        document_id=did,
        chunk_index=0,
        text="hello world",
        chunking_strategy="fixed_size",
        char_count=11,
    )
    assert ch.document_id == did
    assert ch.chunk_index == 0


def test_chunk_invalid_negative_index() -> None:
    with pytest.raises(ValueError, match="chunk_index"):
        Chunk(
            document_id=uuid.uuid4(),
            chunk_index=-1,
            text="x",
            chunking_strategy="fixed_size",
            char_count=1,
        )


def test_processing_run_minimal() -> None:
    run = ProcessingRun(run_type="ingest", status="pending")
    assert run.documents_processed == 0
    assert run.metadata == {}


def test_processing_run_with_timestamps() -> None:
    t = datetime.now(UTC)
    run = ProcessingRun(
        run_type="index",
        status="completed",
        started_at=t,
        finished_at=t,
        documents_processed=2,
        chunks_created=5,
        metadata={"corpus": "sample"},
    )
    assert run.documents_processed == 2


def test_supported_retrieval_modes_unchanged() -> None:
    assert "dense_only" in SUPPORTED_RETRIEVAL_MODES
    assert "sparse_only" in SUPPORTED_RETRIEVAL_MODES
    assert "hybrid" in SUPPORTED_RETRIEVAL_MODES
    assert len(SUPPORTED_RETRIEVAL_MODES) == 3
