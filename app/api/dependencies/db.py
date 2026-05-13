"""FastAPI DB session dependency wrapping session_scope."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.storage.postgres.session import session_scope


def get_session() -> Iterator[Session]:
    """Yield one SQLAlchemy session per request (commit on success, rollback on error)."""
    with session_scope(get_settings().database_url) as session:
        yield session
