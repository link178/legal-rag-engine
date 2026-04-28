"""FastAPI application entrypoint (Phase 1 bootstrap)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import health as health_routes
from app.core.config import get_settings
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    configure_logging(settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health_routes.router)
    # Mount root and /health from the same router (both defined on router)
    return app


app = create_app()
