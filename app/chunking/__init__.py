"""Chunking package: strategies, config, service."""

from __future__ import annotations

from app.chunking.models import ChunkingConfig
from app.chunking.service import ChunkingService, default_chunking_service

__all__ = ["ChunkingConfig", "ChunkingService", "default_chunking_service"]
