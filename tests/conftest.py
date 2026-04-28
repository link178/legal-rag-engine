"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def reset_settings_cache() -> None:
    """Ensure each test sees a fresh cached Settings when env is patched."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
