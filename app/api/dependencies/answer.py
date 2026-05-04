"""Answer route dependencies: reject non-mock provider before DB session."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.api.schemas.answer import AnswerRequest
from app.core.config import get_settings
from app.generation.errors import UnsupportedProviderError
from app.storage.postgres.session import session_scope


def validate_mock_provider(body: AnswerRequest) -> AnswerRequest:
    if body.provider.strip().lower() != "mock":
        raise UnsupportedProviderError(
            f"Only generation provider 'mock' is supported; got {body.provider!r}"
        )
    return body


def get_session_for_answer(
    _validated: AnswerRequest = Depends(validate_mock_provider),
) -> Iterator[Session]:
    with session_scope(get_settings().database_url) as session:
        yield session
