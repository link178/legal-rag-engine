"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest
from app.core.config import get_settings
from app.storage.postgres.base import Base
from app.storage.postgres.session import session_scope
from sqlalchemy import delete


@pytest.fixture(autouse=True)
def reset_settings_cache() -> None:
    """Ensure each test sees a fresh cached Settings when env is patched."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def isolate_integration_db(request: pytest.FixtureRequest) -> None:
    """
    Start each integration DB test from a clean state.

    Applies only when:
    - test is marked ``@pytest.mark.integration``
    - ``LEGAL_RAG_RUN_INTEGRATION_DB=1`` is set
    """
    if request.node.get_closest_marker("integration") is None:
        yield
        return

    if os.environ.get("LEGAL_RAG_RUN_INTEGRATION_DB") != "1":
        yield
        return

    with session_scope() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(delete(table))

    yield
