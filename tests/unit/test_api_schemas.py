"""API Pydantic schemas: bounds, extra=forbid, engine mode literals."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from app.api.schemas.answer import AnswerRequest, AnswerResponse, GroundedCitationResponse
from app.api.schemas.chunk import ChunkRequest, ChunkResponse
from app.api.schemas.documents import DocumentListItem, DocumentListResponse
from app.api.schemas.index import IndexRequest, IndexResponse
from app.api.schemas.ingest import IngestRequest
from app.api.schemas.retrieve import RetrievedChunkResponse, RetrieveRequest, RetrieveResponse
from pydantic import ValidationError


def test_retrieve_request_valid() -> None:
    r = RetrieveRequest(query="hello", mode="hybrid", top_k=5)
    assert r.query == "hello"
    assert r.mode == "hybrid"


def test_retrieve_request_query_stripped() -> None:
    r = RetrieveRequest(query="  x  ")
    assert r.query == "x"


def test_retrieve_request_empty_query_fails() -> None:
    with pytest.raises(ValidationError):
        RetrieveRequest(query="")
    with pytest.raises(ValidationError):
        RetrieveRequest(query="   ")


def test_retrieve_request_top_k_bounds() -> None:
    with pytest.raises(ValidationError):
        RetrieveRequest(query="q", top_k=0)
    with pytest.raises(ValidationError):
        RetrieveRequest(query="q", top_k=51)


def test_retrieve_request_mode_must_be_engine_literal() -> None:
    with pytest.raises(ValidationError):
        RetrieveRequest(query="q", mode="dense")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RetrieveRequest(query="q", mode="sparse")  # type: ignore[arg-type]


def test_retrieve_request_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        RetrieveRequest(query="q", embedding_provider="x")  # type: ignore[call-arg]


def test_answer_request_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        AnswerRequest(question="q", foo=1)  # type: ignore[call-arg]


def test_answer_request_provider_default() -> None:
    a = AnswerRequest(question="why?")
    assert a.provider == "mock"


def test_retrieve_request_metadata_filter_partial() -> None:
    r = RetrieveRequest(
        query="q",
        metadata_filter={"jurisdiction": "eu", "legal_document_type": "regulation"},
    )
    assert r.metadata_filter is not None
    assert r.metadata_filter.jurisdiction == "eu"
    assert r.metadata_filter.legal_document_type == "regulation"


def test_retrieve_request_metadata_filter_strips_whitespace() -> None:
    r = RetrieveRequest(query="q", metadata_filter={"jurisdiction": "  eu  "})
    assert r.metadata_filter is not None
    assert r.metadata_filter.jurisdiction == "eu"


def test_retrieve_request_metadata_filter_empty_subobject() -> None:
    r = RetrieveRequest(query="q", metadata_filter={})
    assert r.metadata_filter is not None
    assert r.metadata_filter.jurisdiction is None


def test_retrieve_request_metadata_filter_unknown_nested_key() -> None:
    with pytest.raises(ValidationError):
        RetrieveRequest(query="q", metadata_filter={"not_a_field": "x"})  # type: ignore[arg-type]


def test_answer_request_metadata_filter() -> None:
    a = AnswerRequest(question="q?", metadata_filter={"language": "en"})
    assert a.metadata_filter is not None
    assert a.metadata_filter.language == "en"


def test_retrieval_config_from_params_maps_metadata_filter() -> None:
    from app.api.retrieval_config import retrieval_config_from_params

    class _S:
        embedding_provider = "deterministic_hash"
        embedding_dimensions = 16
        embedding_model = ""

    p = RetrieveRequest(
        query="q",
        metadata_filter={"jurisdiction": "eu"},
    )
    cfg = retrieval_config_from_params(p, _S())  # type: ignore[arg-type]
    assert cfg.metadata_filter is not None
    assert cfg.metadata_filter.jurisdiction == "eu"
    assert cfg.metadata_filter.as_dict() == {"jurisdiction": "eu"}


def test_retrieval_config_from_params_empty_metadata_filter_object() -> None:
    from app.api.retrieval_config import retrieval_config_from_params

    class _S:
        embedding_provider = "deterministic_hash"
        embedding_dimensions = 16
        embedding_model = ""

    p = RetrieveRequest(query="q", metadata_filter={})
    cfg = retrieval_config_from_params(p, _S())  # type: ignore[arg-type]
    assert cfg.metadata_filter is None


def test_ingest_request_path_required() -> None:
    with pytest.raises(ValidationError):
        IngestRequest(path="")


def test_retrieve_response_roundtrip() -> None:
    cid = uuid4()
    did = uuid4()
    chunk = RetrievedChunkResponse(
        chunk_id=cid,
        document_id=did,
        text_preview="hi",
        chunk_index=0,
        chunking_strategy="fixed_size",
        rank=1,
    )
    out = RetrieveResponse(
        query="q",
        mode="hybrid",
        top_k=5,
        total_results=1,
        chunks=[chunk],
    )
    d = out.model_dump(mode="json")
    assert d["total_results"] == 1
    assert d["chunks"][0]["chunk_id"] == str(cid)


def test_answer_response_roundtrip() -> None:
    ar = AnswerResponse(
        question="q",
        answer="a",
        mode="grounded",
        insufficient_context=False,
        retrieval_mode="hybrid",
        citations=[
            GroundedCitationResponse(
                citation_id=1,
                chunk_id=str(uuid4()),
                document_id=str(uuid4()),
                rank=1,
                score=0.5,
                text_preview="x",
            )
        ],
        used_citation_ids=[1],
    )
    assert ar.mode == "grounded"
    json.loads(ar.model_dump_json())


def test_document_list_response() -> None:
    from datetime import UTC, datetime

    doc_id = uuid4()
    lr = DocumentListResponse(
        documents=[
            DocumentListItem(
                id=doc_id,
                source_path="p.md",
                checksum="c",
                created_at=datetime.now(UTC),
            )
        ],
        count=1,
        limit=50,
        offset=0,
    )
    assert lr.count == 1


def test_chunk_request_overlap_must_be_below_size() -> None:
    with pytest.raises(ValidationError):
        ChunkRequest(
            document_id=uuid4(),
            chunk_size=100,
            chunk_overlap=100,
        )


def test_chunk_request_defaults() -> None:
    cid = uuid4()
    req = ChunkRequest(document_id=cid)
    assert req.strategy == "fixed_size"
    assert req.chunk_size == 1200
    assert req.chunk_overlap == 200


def test_index_request_dense_sparse_both_false() -> None:
    with pytest.raises(ValidationError):
        IndexRequest(include_dense=False, include_sparse=False)


def test_index_request_embedding_dimensions_zero() -> None:
    with pytest.raises(ValidationError):
        IndexRequest(embedding_dimensions=0)


def test_chunk_response_roundtrip() -> None:
    did = uuid4()
    out = ChunkResponse(
        document_id=did,
        processing_run_id=uuid4(),
        strategy="fixed_size",
        config_hash="a" * 64,
        created=True,
        skipped_existing=False,
        chunks_count=1,
        chunk_ids=[uuid4()],
        chunks=[],
        metadata={"k": 1},
    )
    json.loads(out.model_dump_json())


def test_index_response_roundtrip() -> None:
    out = IndexResponse(
        manifest_id=uuid4(),
        manifest_hash="m" * 64,
        config_hash="c" * 64,
        chunk_set_hash="s" * 64,
        chunking_strategy="fixed_size",
        embedding_provider="deterministic_hash",
        embedding_model=None,
        embedding_dimensions=16,
        include_dense=True,
        include_sparse=True,
        embeddings_persisted=True,
        chunk_count=3,
        indexed_chunk_count=3,
        failed_chunks_count=0,
        processing_run_id=uuid4(),
        created=True,
        skipped_existing=False,
        metadata={},
    )
    json.loads(out.model_dump_json())
