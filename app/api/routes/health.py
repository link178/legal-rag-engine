"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    """Basic liveness probe; uses settings only (no DB check in Phase 1)."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
    }


@router.get("/")
def root(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    """Minimal service metadata."""
    return {
        "service": settings.app_name,
        "status": "running",
        "docs": "/docs",
    }
