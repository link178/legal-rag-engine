"""Chunking configuration and deterministic config hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Validated parameters for chunking strategies."""

    strategy: str = "fixed_size"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    min_chunk_chars: int = 80
    preserve_headings: bool = True

    def __post_init__(self) -> None:
        if not self.strategy.strip():
            raise ValueError("strategy must be non-empty")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        if self.min_chunk_chars < 0:
            raise ValueError("min_chunk_chars must be >= 0")

    def normalized_dict(self) -> dict[str, int | bool | str]:
        """Canonical dict for hashing and metadata (sorted keys in JSON)."""
        return {
            "strategy": self.strategy.strip(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_chars": self.min_chunk_chars,
            "preserve_headings": self.preserve_headings,
        }

    def config_hash(self) -> str:
        """SHA-256 hex of canonical JSON (sorted keys)."""
        payload = json.dumps(self.normalized_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
