"""Deterministic SHA-256 checksums for text and files."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_text_checksum(text: str) -> str:
    """SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_file_checksum(path: Path, chunk_size: int = 65536) -> str:
    """SHA-256 hex digest of raw file bytes (streaming, avoids loading huge files at once)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
