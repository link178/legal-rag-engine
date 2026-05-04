"""Answer API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.schemas.retrieve import RetrievalParams


class AnswerRequest(RetrievalParams):
    """POST /v1/answer body (retrieval params + question + generation)."""

    question: str = Field(..., min_length=1)
    provider: str = Field(default="mock")
    max_chunks: int = Field(default=5, ge=1, le=20)
    max_context_chars: int = Field(default=8000, gt=0)
    max_chunk_chars: int = Field(default=2000, gt=0)
    min_score: float | None = None

    @field_validator("question", mode="after")
    @classmethod
    def strip_question(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("question must be non-empty")
        return s


class GroundedCitationResponse(BaseModel):
    """Citation row in answer response."""

    model_config = ConfigDict(extra="forbid")

    citation_id: int
    chunk_id: str
    document_id: str
    source_path: str | None = None
    title: str | None = None
    heading: str | None = None
    rank: int
    score: float | None = None
    text_preview: str


class CitationVerificationResponse(BaseModel):
    """Mechanical citation verification (Phase 7)."""

    model_config = ConfigDict(extra="forbid")

    used_citation_ids: list[int]
    available_citation_ids: list[int]
    valid_citation_ids: list[int]
    invalid_citation_ids: list[int]
    unused_citation_ids: list[int]
    duplicate_citation_ids: list[int]
    citation_validity_rate: float
    has_citations: bool
    has_valid_citations: bool
    has_invalid_citations: bool


class AnswerResponse(BaseModel):
    """POST /v1/answer response (mirrors generation CLI answer_to_dict)."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    mode: Literal["grounded", "partial", "insufficient_context"]
    insufficient_context: bool
    retrieval_mode: str | None = None
    citations: list[GroundedCitationResponse]
    used_citation_ids: list[int]
    citation_verification: CitationVerificationResponse | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
