"""Unit-test isolation: env vars should not override explicit Settings() tests."""

from __future__ import annotations

import pytest

_SETTINGS_ENV_KEYS = (
    "APP_NAME",
    "APP_ENV",
    "DEBUG",
    "DATABASE_URL",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "GENERATION_PROVIDER",
    "LOCAL_LLM_MODEL",
    "RETRIEVAL_MODE",
    "LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def _clear_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _SETTINGS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
