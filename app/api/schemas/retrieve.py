"""Retrieve API schemas (shared retrieval params with answer)."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_PREVIEW_LEN = 240


class MetadataFilterParams(BaseModel):
    """Optional exact-match filters on ``documents.metadata_json`` (Phase 14)."""

    model_config = ConfigDict(extra="forbid")

    corpus_name: str | None = None
    corpus_adapter: str | None = None
    source_family: str | None = None
    jurisdiction: str | None = None
    legal_document_type: str | None = None
    language: str | None = None
    canonical_id: str | None = None

    @model_validator(mode="after")
    def _strip_optional_strings(self) -> MetadataFilterParams:
        updates: dict[str, str | None] = {}
        for name in (
            "corpus_name",
            "corpus_adapter",
            "source_family",
            "jurisdiction",
            "legal_document_type",
            "language",
            "canonical_id",
        ):
            v = getattr(self, name)
            if v is None:
                continue
            s = str(v).strip()
            updates[name] = s if s else None
        return self.model_copy(update=updates)


class RetrievalParams(BaseModel):
    """Fields shared by retrieve and answer (no query/question)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["dense_only", "sparse_only", "hybrid"] = "hybrid"
    chunking_strategy: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    dense_top_k: int = Field(default=10, ge=1, le=50)
    sparse_top_k: int = Field(default=10, ge=1, le=50)
    rrf_k: int = Field(default=60, gt=0)
    index_manifest_id: UUID | None = None
    metadata_filter: MetadataFilterParams | None = None

    @field_validator("chunking_strategy", mode="before")
    @classmethod
    def normalize_chunking_strategy(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None


class RetrieveRequest(RetrievalParams):
    """POST /v1/retrieve body."""

    query: str = Field(..., min_length=1)

    @field_validator("query", mode="after")
    @classmethod
    def strip_query(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("query must be non-empty")
        return s


class RetrievedChunkResponse(BaseModel):
    """One hit in retrieve response (matches CLI / RetrievedChunk)."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: UUID
    document_id: UUID
    text_preview: str
    source_path: str | None = None
    title: str | None = None
    heading: str | None = None
    chunk_index: int
    chunking_strategy: str
    rank: int
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    retrieval_sources: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    """POST /v1/retrieve response (mirrors retrieval CLI JSON)."""

    model_config = ConfigDict(extra="forbid")

    query: str
    mode: str
    top_k: int
    index_manifest_id: UUID | None = None
    manifest_hash: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    total_results: int
    chunks: list[RetrievedChunkResponse]
    metadata: dict[str, Any] = Field(default_factory=dict)


def text_preview(text: str, *, max_len: int = _PREVIEW_LEN) -> str:
    """Match app.retrieval.cli _PREVIEW_LEN truncation."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "\u2026"
