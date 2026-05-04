"""Deterministic generator for tests and zero-cost operator demos."""

from __future__ import annotations

import re

from app.generation.providers.base import INSUFFICIENT_CONTEXT_SENTENCE

_BLOCK_HEADER = re.compile(r"^\[(\d+)\]\s", re.MULTILINE)


class MockGenerationProvider:
    """Parse prompt block headers and emit a fixed answer citing up to three ids."""

    name = "mock"

    def generate(self, prompt: str) -> str:
        ids: list[int] = []
        seen: set[int] = set()
        for m in _BLOCK_HEADER.finditer(prompt):
            n = int(m.group(1))
            if n not in seen:
                seen.add(n)
                ids.append(n)
        if not ids:
            return INSUFFICIENT_CONTEXT_SENTENCE
        take = ids[:3]
        refs = "".join(f"[{i}]" for i in take)
        return (
            "Based on the provided context, the answer is supported by the "
            f"retrieved evidence. {refs}"
        )
