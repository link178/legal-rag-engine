"""Checksum utilities."""

from __future__ import annotations

import hashlib

from app.ingestion.checksums import compute_file_checksum, compute_text_checksum


def test_compute_text_checksum_deterministic() -> None:
    a = compute_text_checksum("hello")
    b = compute_text_checksum("hello")
    assert a == b
    assert a == hashlib.sha256(b"hello").hexdigest()


def test_compute_file_checksum_changes_with_content(tmp_path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("one", encoding="utf-8")
    c1 = compute_file_checksum(p)
    p.write_text("two", encoding="utf-8")
    c2 = compute_file_checksum(p)
    assert c1 != c2
