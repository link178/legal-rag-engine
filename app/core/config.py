"""Central application settings loaded from environment (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import DEFAULT_RETRIEVAL_MODE, SUPPORTED_RETRIEVAL_MODES


class Settings(BaseSettings):
    """Bootstrap settings; extend in later phases for chunking, RRF, etc."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Python field names in Settings(...); env still uses aliases (RETRIEVAL_MODE, ...).
        populate_by_name=True,
    )

    app_name: str = Field(default="legal-rag-engine", validation_alias="APP_NAME")
    app_env: Literal["development", "test", "staging", "production"] = Field(
        default="development",
        validation_alias="APP_ENV",
    )
    debug: bool = Field(default=True, validation_alias="DEBUG")

    database_url: str = Field(
        default="postgresql+psycopg://legal_rag:legal_rag@localhost:5432/legal_rag",
        validation_alias="DATABASE_URL",
    )

    embedding_provider: str = Field(
        default="deterministic_hash", validation_alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str = Field(default="", validation_alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=16, validation_alias="EMBEDDING_DIMENSIONS")
    generation_provider: str = Field(default="mock", validation_alias="GENERATION_PROVIDER")
    local_llm_model: str = Field(default="", validation_alias="LOCAL_LLM_MODEL")

    retrieval_mode: str = Field(
        default=DEFAULT_RETRIEVAL_MODE,
        validation_alias="RETRIEVAL_MODE",
    )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @field_validator("retrieval_mode")
    @classmethod
    def validate_retrieval_mode(cls, v: str) -> str:
        if v not in SUPPORTED_RETRIEVAL_MODES:
            raise ValueError(
                f"RETRIEVAL_MODE must be one of {SUPPORTED_RETRIEVAL_MODES}, got {v!r}"
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """Return cached settings (invalidate in tests via get_settings.cache_clear())."""
    return Settings()
