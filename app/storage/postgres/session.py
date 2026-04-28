"""Engine and session factory. No connections at import time."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


@lru_cache(maxsize=16)
def _engine_for_url(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


def get_engine(database_url: str | None = None) -> Engine:
    """Return a cached SQLAlchemy engine for ``database_url`` or ``Settings.database_url``."""
    if database_url is None:
        from app.core.config import get_settings

        database_url = get_settings().database_url
    return _engine_for_url(database_url)


def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    """Return a ``sessionmaker`` bound to the engine for the given URL."""
    engine = get_engine(database_url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


@contextmanager
def session_scope(database_url: str | None = None) -> Iterator[Session]:
    """Yield a session that commits on success and rolls back on error."""
    factory = get_session_factory(database_url)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def invalidate_engine_cache() -> None:
    """Clear cached engines (for tests that swap ``DATABASE_URL``)."""
    _engine_for_url.cache_clear()
