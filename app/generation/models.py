"""Grounded generation result types (pure; frozen dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

AnswerMode = Literal["grounded", "partial", "insufficient_context"]

SUPPORTED_ANSWER_MODES: tuple[str, ...] = ("grounded", "partial", "insufficient_context")


def _empty_metadata() -> dict[str, Any]:
    return {}


_PREVIEW_LEN = 240


@dataclass(frozen=True, kw_only=True, slots=True)
class CitationVerificationResult:
    """Mechanical verification of bracketed citation ids vs context blocks."""

    used_citation_ids: tuple[int, ...]
    available_citation_ids: tuple[int, ...]
    valid_citation_ids: tuple[int, ...]
    invalid_citation_ids: tuple[int, ...]
    unused_citation_ids: tuple[int, ...]
    duplicate_citation_ids: tuple[int, ...]
    citation_validity_rate: float
    has_citations: bool
    has_valid_citations: bool
    has_invalid_citations: bool

    def __post_init__(self) -> None:
        r = self.citation_validity_rate
        if r < 0.0 or r > 1.0 or not (r == r):  # reject NaN
            raise ValueError(
                "citation_validity_rate must be finite and in [0.0, 1.0]; "
                f"got {self.citation_validity_rate!r}"
            )


def empty_verification(*, available: tuple[int, ...] = ()) -> CitationVerificationResult:
    """No brackets used in the answer; optional available ids from context."""
    avail = tuple(sorted(dict.fromkeys(available)))
    return CitationVerificationResult(
        used_citation_ids=(),
        available_citation_ids=avail,
        valid_citation_ids=(),
        invalid_citation_ids=(),
        unused_citation_ids=avail,
        duplicate_citation_ids=(),
        citation_validity_rate=0.0,
        has_citations=False,
        has_valid_citations=False,
        has_invalid_citations=False,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class GroundedContextBlock:
    """One citable snippet passed to the prompt builder."""

    citation_id: int
    chunk_id: UUID
    document_id: UUID
    source_path: str | None
    title: str | None
    heading: str | None
    rank: int
    score: float | None
    text: str

    def __post_init__(self) -> None:
        if self.citation_id < 1:
            raise ValueError("citation_id must be >= 1")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if not self.text.strip():
            raise ValueError("text must be non-empty after strip")


@dataclass(frozen=True, kw_only=True, slots=True)
class GroundedCitation:
    """Citation metadata surfaced in CLI / JSON (subset of GroundedContextBlock)."""

    citation_id: int
    chunk_id: UUID
    document_id: UUID
    source_path: str | None
    title: str | None
    heading: str | None
    rank: int
    score: float | None
    text_preview: str

    def __post_init__(self) -> None:
        if self.citation_id < 1:
            raise ValueError("citation_id must be >= 1")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if not self.text_preview:
            raise ValueError("text_preview must be non-empty")


@dataclass(frozen=True, kw_only=True, slots=True)
class GroundedAnswer:
    """Structured answer after retrieval + grounding + generation step."""

    question: str
    answer: str
    mode: str
    citations: tuple[GroundedCitation, ...]
    used_citation_ids: tuple[int, ...]
    retrieval_mode: str | None
    insufficient_context: bool
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)
    citation_verification: CitationVerificationResult | None = None

    def __post_init__(self) -> None:
        q = self.question.strip()
        if not q:
            raise ValueError("question must be non-empty")
        object.__setattr__(self, "question", q)

        a = self.answer.strip()
        if not a:
            raise ValueError("answer must be non-empty")
        object.__setattr__(self, "answer", a)

        if self.mode not in SUPPORTED_ANSWER_MODES:
            raise ValueError(
                f"mode must be one of {SUPPORTED_ANSWER_MODES}, got {self.mode!r}"
            )

        insuff = self.mode == "insufficient_context"
        if self.insufficient_context != insuff:
            raise ValueError(
                "insufficient_context must be True when mode is insufficient_context, "
                "False otherwise"
            )

        if self.mode == "grounded":
            if not self.citations:
                raise ValueError("grounded mode requires at least one citation")
            if not self.used_citation_ids:
                raise ValueError("grounded mode requires at least one used_citation_id")
            cited = {c.citation_id for c in self.citations}
            for uid in self.used_citation_ids:
                if uid not in cited:
                    raise ValueError(
                        f"used_citation_id {uid} has no matching citation in citations"
                    )
        elif self.mode == "insufficient_context":
            if self.citations:
                raise ValueError("insufficient_context mode must have empty citations")
            if self.used_citation_ids:
                raise ValueError("insufficient_context mode must have empty used_citation_ids")

        cv = self.citation_verification
        if cv is not None:
            if self.mode == "grounded":
                if not cv.has_valid_citations or cv.has_invalid_citations:
                    raise ValueError(
                        "grounded mode requires citation_verification with "
                        "has_valid_citations=True and has_invalid_citations=False"
                    )
            elif self.mode == "insufficient_context":
                if cv.has_citations:
                    raise ValueError(
                        "insufficient_context mode requires citation_verification "
                        "with has_citations=False"
                    )


def citation_from_block(block: GroundedContextBlock) -> GroundedCitation:
    """Build a citation row with deterministic text preview length."""
    if len(block.text) <= _PREVIEW_LEN:
        preview = block.text
    else:
        preview = block.text[: _PREVIEW_LEN - 1] + "\u2026"
    return GroundedCitation(
        citation_id=block.citation_id,
        chunk_id=block.chunk_id,
        document_id=block.document_id,
        source_path=block.source_path,
        title=block.title,
        heading=block.heading,
        rank=block.rank,
        score=block.score,
        text_preview=preview,
    )
