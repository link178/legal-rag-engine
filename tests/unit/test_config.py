"""Unit tests for application settings (no database)."""

from __future__ import annotations

import pytest
from app.core.config import Settings, get_settings
from app.core.constants import DEFAULT_RETRIEVAL_MODE, SUPPORTED_RETRIEVAL_MODES
from pydantic import ValidationError


def test_defaults_match_bootstrap() -> None:
    s = Settings(_env_file=None)
    assert s.app_name == "legal-rag-engine"
    assert s.app_env == "development"
    assert s.debug is True
    assert "postgresql" in s.database_url or s.database_url.startswith("postgresql")
    assert s.embedding_provider == "local"
    assert s.generation_provider == "mock"
    assert s.retrieval_mode == DEFAULT_RETRIEVAL_MODE
    assert s.log_level == "INFO"


def test_get_settings_cached() -> None:
    a = get_settings()
    b = get_settings()
    assert a is b


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_NAME", "custom-service")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    s = Settings(_env_file=None)
    assert s.app_name == "custom-service"
    assert s.app_env == "test"
    assert s.debug is False
    assert s.log_level == "DEBUG"


def test_retrieval_mode_invalid() -> None:
    with pytest.raises(ValidationError, match="RETRIEVAL_MODE"):
        Settings(retrieval_mode="invalid", _env_file=None)  # type: ignore[arg-type]


def test_retrieval_mode_all_supported() -> None:
    for mode in SUPPORTED_RETRIEVAL_MODES:
        s = Settings(retrieval_mode=mode, _env_file=None)
        assert s.retrieval_mode == mode


def test_unknown_env_var_does_not_break_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNKNOWN_FUTURE_VAR", "should_not_break")
    s = Settings(_env_file=None)
    assert "unknown_future_var" not in s.model_dump()
