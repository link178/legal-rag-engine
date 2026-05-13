"""FastAPI application entrypoint (Phase 1 bootstrap + Phase 8 API v1)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.routes import answer as answer_routes
from app.api.routes import chunk as chunk_routes
from app.api.routes import documents as documents_routes
from app.api.routes import health as health_routes
from app.api.routes import index as index_routes
from app.api.routes import ingest as ingest_routes
from app.api.routes import retrieve as retrieve_routes
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
    register_exception_handlers(app)
    app.include_router(health_routes.router)
    app.include_router(ingest_routes.router, prefix="/v1", tags=["ingest"])
    app.include_router(documents_routes.router, prefix="/v1", tags=["documents"])
    app.include_router(chunk_routes.router, prefix="/v1", tags=["chunk"])
    app.include_router(index_routes.router, prefix="/v1", tags=["index"])
    app.include_router(retrieve_routes.router, prefix="/v1", tags=["retrieve"])
    app.include_router(answer_routes.router, prefix="/v1", tags=["answer"])
    return app


app = create_app()
