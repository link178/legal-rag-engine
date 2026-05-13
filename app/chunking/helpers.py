"""Shared helpers for chunk construction (pure functions)."""

from __future__ import annotations

import math
from typing import Any

from app.ingestion.checksums import compute_text_checksum


def estimate_tokens(char_count: int) -> int:
    """Rough token estimate: ceil(chars / 4), minimum 1."""
    return max(1, math.ceil(char_count / 4))


def chunk_checksum(text: str) -> str:
    """Deterministic SHA-256 hex for chunk body."""
    return compute_text_checksum(text)


def merge_chunk_metadata(
    base: dict[str, Any],
    *,
    start_offset: int,
    end_offset: int,
    chunking_config_hash: str,
    chunking_config: dict[str, Any],
    source_document_checksum: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Standard trace fields on every chunk metadata."""
    out: dict[str, Any] = {
        **base,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "chunking_config_hash": chunking_config_hash,
        "chunking_config": chunking_config,
        "source_document_checksum": source_document_checksum,
    }
    if extra:
        out.update(extra)
    return out
