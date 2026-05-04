"""``SentenceTransformersEmbeddingProvider`` with faked ``sentence_transformers`` (no network)."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

import pytest
from app.indexing.dense.local_provider import SentenceTransformersEmbeddingProvider


def test_init_fails_with_clear_message_when_dependency_missing() -> None:
    real_import = __import__

    def deny_st(name: str, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=deny_st):
        with pytest.raises(RuntimeError, match=r"pip install -e '\.\[local-embeddings\]'"):
            SentenceTransformersEmbeddingProvider(model_name="fake/model", dimensions=3)


def test_lazy_model_load_and_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, int] = {"construct": 0, "encode": 0}

    class FakeModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 4

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            calls["encode"] += 1

            class Row:
                def __init__(self, vals: list[float]):
                    self._vals = vals

                def tolist(self) -> list[float]:
                    return self._vals

            return [Row([1.0, 0.0, 0.0, 0.0]) for _ in texts]

    def sentence_transformer_factory(name: str):
        calls["construct"] += 1
        assert name == "fake/model"
        return FakeModel()

    fake_mod = ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = sentence_transformer_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

    p = SentenceTransformersEmbeddingProvider(model_name="fake/model", dimensions=4)
    assert calls["construct"] == 0
    v = p.embed_query("hello")
    assert calls["construct"] == 1
    assert calls["encode"] == 1
    assert len(v) == 4
    batch = p.embed_texts(["a", "b"])
    assert calls["encode"] == 2
    assert len(batch) == 2


def test_dimension_mismatch_raises_on_first_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 99

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            class Row:
                def tolist(self) -> list[float]:
                    return [0.0] * 99

            return [Row() for _ in texts]

    fake_mod = ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = lambda name: FakeModel()  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

    p = SentenceTransformersEmbeddingProvider(model_name="m", dimensions=4)
    with pytest.raises(ValueError, match="outputs dimension 99"):
        p.embed_query("x")
