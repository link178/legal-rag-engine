"""Map engine types to API response models."""

from __future__ import annotations

from typing import Literal, cast
from uuid import UUID

from app.api.schemas.answer import (
    AnswerResponse,
    CitationVerificationResponse,
    GroundedCitationResponse,
)
from app.api.schemas.chunk import ChunkItemResponse, ChunkResponse
from app.api.schemas.index import (
    IndexManifestDetailResponse,
    IndexManifestItemResponse,
    IndexResponse,
)
from app.api.schemas.retrieve import RetrievedChunkResponse, RetrieveResponse, text_preview
from app.chunking.runner import ChunkingRunResult
from app.domain.models import Chunk
from app.generation.models import CitationVerificationResult, GroundedAnswer, GroundedCitation
from app.indexing.models import IndexingRunResult
from app.retrieval.models import RetrievalResultSet, RetrievedChunk
from app.storage.postgres.models import IndexManifestRecord


def retrieved_chunk_to_response(r: RetrievedChunk) -> RetrievedChunkResponse:
    return RetrievedChunkResponse(
        chunk_id=r.chunk_id,
        document_id=r.document_id,
        text_preview=text_preview(r.text),
        source_path=r.source_path,
        title=r.title,
        heading=r.heading,
        chunk_index=r.chunk_index,
        chunking_strategy=r.chunking_strategy,
        rank=r.rank_position,
        dense_score=r.dense_score,
        sparse_score=r.sparse_score,
        rrf_score=r.rrf_score,
        retrieval_sources=list(r.retrieval_sources),
        metadata=dict(r.metadata),
    )


def retrieval_result_to_response(res: RetrievalResultSet, top_k: int) -> RetrieveResponse:
    return RetrieveResponse(
        query=res.query,
        mode=res.mode,
        top_k=top_k,
        index_manifest_id=res.index_manifest_id,
        manifest_hash=res.manifest_hash,
        embedding_provider=res.embedding_provider,
        embedding_model=res.embedding_model,
        embedding_dimensions=res.embedding_dimensions,
        total_results=len(res.results),
        chunks=[retrieved_chunk_to_response(x) for x in res.results],
        metadata=dict(res.metadata),
    )


def _verification_to_response(v: CitationVerificationResult) -> CitationVerificationResponse:
    return CitationVerificationResponse(
        used_citation_ids=list(v.used_citation_ids),
        available_citation_ids=list(v.available_citation_ids),
        valid_citation_ids=list(v.valid_citation_ids),
        invalid_citation_ids=list(v.invalid_citation_ids),
        unused_citation_ids=list(v.unused_citation_ids),
        duplicate_citation_ids=list(v.duplicate_citation_ids),
        citation_validity_rate=v.citation_validity_rate,
        has_citations=v.has_citations,
        has_valid_citations=v.has_valid_citations,
        has_invalid_citations=v.has_invalid_citations,
    )


def _citation_to_response(c: GroundedCitation) -> GroundedCitationResponse:
    return GroundedCitationResponse(
        citation_id=c.citation_id,
        chunk_id=str(c.chunk_id),
        document_id=str(c.document_id),
        source_path=c.source_path,
        title=c.title,
        heading=c.heading,
        rank=c.rank,
        score=c.score,
        text_preview=c.text_preview,
    )


def grounded_answer_to_response(answer: GroundedAnswer) -> AnswerResponse:
    cv = (
        _verification_to_response(answer.citation_verification)
        if answer.citation_verification is not None
        else None
    )
    mode = cast(Literal["grounded", "partial", "insufficient_context"], answer.mode)
    return AnswerResponse(
        question=answer.question,
        answer=answer.answer,
        mode=mode,
        insufficient_context=answer.insufficient_context,
        retrieval_mode=answer.retrieval_mode,
        citations=[_citation_to_response(c) for c in answer.citations],
        used_citation_ids=list(answer.used_citation_ids),
        citation_verification=cv,
        metadata=dict(answer.metadata),
    )


def chunk_record_to_response(
    c: Chunk,
    *,
    include_text: bool,
    preview_chars: int,
) -> ChunkItemResponse:
    if c.id is None:
        raise ValueError("chunk id is required for API response")
    tp = text_preview(c.text, max_len=preview_chars) if include_text else ""
    return ChunkItemResponse(
        chunk_id=c.id,
        document_id=c.document_id,
        chunk_index=c.chunk_index,
        chunking_strategy=c.chunking_strategy,
        char_count=c.char_count,
        heading=c.heading,
        page_number=c.page_number,
        checksum=c.checksum,
        metadata=dict(c.metadata),
        text_preview=tp,
    )


def chunking_run_result_to_response(
    r: ChunkingRunResult,
    *,
    document_id: UUID,
    config_hash: str,
    total_chunks: int,
    persisted_chunks: list[Chunk] | None,
    include_chunks: bool,
    preview_chars: int,
) -> ChunkResponse:
    run_meta = dict(r.run.metadata) if r.run and r.run.metadata else {}
    chunks_out: list[ChunkItemResponse] = []
    if include_chunks and persisted_chunks:
        chunks_out = [
            chunk_record_to_response(
                ch, include_text=True, preview_chars=preview_chars
            )
            for ch in persisted_chunks
        ]
    run_id = r.run.id if r.run and r.run.id else None
    created = r.created_chunks > 0
    return ChunkResponse(
        document_id=document_id,
        processing_run_id=run_id,
        strategy=r.strategy,
        config_hash=config_hash,
        created=created,
        skipped_existing=r.skipped_existing,
        chunks_count=total_chunks,
        chunk_ids=list(r.chunk_ids),
        chunks=chunks_out,
        metadata=run_meta,
    )


def indexing_run_result_to_response(
    r: IndexingRunResult,
    *,
    fallback_provider: str,
    fallback_dims: int,
    fallback_model: str | None,
) -> IndexResponse:
    failed = sum(1 for x in r.chunk_results if x.error is not None)
    m = r.manifest
    run_meta = dict(r.run.metadata) if r.run and r.run.metadata else {}
    run_id = r.run.id if r.run and r.run.id else None
    created = not r.skipped_existing

    if m is None:
        return IndexResponse(
            manifest_id=None,
            manifest_hash=None,
            config_hash=None,
            chunk_set_hash=None,
            chunking_strategy=None,
            embedding_provider=fallback_provider,
            embedding_model=fallback_model,
            embedding_dimensions=fallback_dims,
            include_dense=True,
            include_sparse=True,
            embeddings_persisted=False,
            chunk_count=0,
            indexed_chunk_count=0,
            failed_chunks_count=failed,
            processing_run_id=run_id,
            created=False,
            skipped_existing=r.skipped_existing,
            metadata=run_meta,
        )

    return IndexResponse(
        manifest_id=m.id,
        manifest_hash=m.manifest_hash,
        config_hash=m.config_hash,
        chunk_set_hash=m.chunk_set_hash,
        chunking_strategy=m.chunking_strategy,
        embedding_provider=m.embedding_provider,
        embedding_model=m.embedding_model,
        embedding_dimensions=m.embedding_dimensions,
        include_dense=m.include_dense,
        include_sparse=m.include_sparse,
        embeddings_persisted=m.embeddings_persisted,
        chunk_count=m.chunk_count,
        indexed_chunk_count=m.indexed_chunk_count,
        failed_chunks_count=failed,
        processing_run_id=run_id,
        created=created,
        skipped_existing=r.skipped_existing,
        metadata=run_meta,
    )


def index_manifest_record_to_item(row: IndexManifestRecord) -> IndexManifestItemResponse:
    return IndexManifestItemResponse(
        manifest_id=row.id,
        manifest_hash=row.manifest_hash,
        config_hash=row.config_hash,
        chunk_set_hash=row.chunk_set_hash,
        chunking_strategy=row.chunking_strategy,
        embedding_provider=row.embedding_provider,
        embedding_model=row.embedding_model,
        embedding_dimensions=row.embedding_dimensions,
        include_dense=row.include_dense,
        include_sparse=row.include_sparse,
        embeddings_persisted=row.embeddings_persisted,
        document_count=row.document_count,
        chunk_count=row.chunk_count,
        indexed_chunk_count=row.indexed_chunk_count,
        corpus_version=row.corpus_version,
        created_at=row.created_at,
        updated_at=row.updated_at,
        metadata=dict(row.metadata_json),
    )


def index_manifest_record_to_detail(row: IndexManifestRecord) -> IndexManifestDetailResponse:
    base = index_manifest_record_to_item(row)
    return IndexManifestDetailResponse.model_validate(base.model_dump(mode="python"))
