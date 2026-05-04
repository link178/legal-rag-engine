"""POST /v1/answer — read-only; mock provider only."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies.answer import get_session_for_answer, validate_mock_provider
from app.api.mappers import grounded_answer_to_response
from app.api.retrieval_config import retrieval_config_from_params
from app.api.schemas.answer import AnswerRequest, AnswerResponse
from app.core.config import Settings, get_settings
from app.generation.answerer import GroundedAnswerer
from app.generation.context import ContextBuilder
from app.generation.providers import MockGenerationProvider

router = APIRouter()


@router.post("/answer", response_model=AnswerResponse)
def answer_question(
    body: AnswerRequest = Depends(validate_mock_provider),
    session: Session = Depends(get_session_for_answer),
    settings: Settings = Depends(get_settings),
) -> AnswerResponse:
    cfg = retrieval_config_from_params(body, settings)
    try:
        ctx_builder = ContextBuilder(
            max_chunks=body.max_chunks,
            max_context_chars=body.max_context_chars,
            max_chunk_chars=body.max_chunk_chars,
            min_score=body.min_score,
        )
    except ValueError as e:
        raise ValueError(f"context builder: {e}") from e

    ga = GroundedAnswerer.from_session(
        session,
        cfg,
        context_builder=ctx_builder,
        provider=MockGenerationProvider(),
    )
    out = ga.answer(body.question)
    return grounded_answer_to_response(out)
