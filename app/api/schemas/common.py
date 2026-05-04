"""Shared API envelope types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiErrorBody(BaseModel):
    """Single error object inside ApiErrorResponse."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ApiErrorResponse(BaseModel):
    """Standard error JSON envelope."""

    model_config = ConfigDict(extra="forbid")

    error: ApiErrorBody
