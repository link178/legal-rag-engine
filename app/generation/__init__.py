"""Grounded generation: context building, mock provider, grounded answer orchestration."""

from __future__ import annotations

from app.generation.answerer import GroundedAnswerer
from app.generation.citations import extract_cited_ids
from app.generation.context import ContextBuilder
from app.generation.errors import GenerationError, UnsupportedProviderError
from app.generation.models import (
    AnswerMode,
    GroundedAnswer,
    GroundedCitation,
    GroundedContextBlock,
    citation_from_block,
)
from app.generation.prompts import build_grounded_prompt
from app.generation.providers import (
    INSUFFICIENT_CONTEXT_SENTENCE,
    GenerationProvider,
    MockGenerationProvider,
)

__all__ = [
    "AnswerMode",
    "ContextBuilder",
    "GenerationError",
    "GenerationProvider",
    "GroundedAnswer",
    "GroundedAnswerer",
    "GroundedCitation",
    "GroundedContextBlock",
    "INSUFFICIENT_CONTEXT_SENTENCE",
    "MockGenerationProvider",
    "UnsupportedProviderError",
    "build_grounded_prompt",
    "citation_from_block",
    "extract_cited_ids",
]
