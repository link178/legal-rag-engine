"""Optional real embeddings via sentence-transformers (lazy import, lazy model load)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_INSTALL_HINT = "Install local embeddings extra: pip install -e '.[local-embeddings]'"


class SentenceTransformersEmbeddingProvider:
    """
    Local dense embeddings using ``sentence_transformers.SentenceTransformer``.

    Dependencies are imported only in ``__init__`` (not at module import time).
    The underlying model is loaded on the first call to ``embed_texts`` or ``embed_query``
    to keep construction cheap and avoid downloads during module import.

    Embeddings are L2-normalized when ``normalize_embeddings`` is True (default), consistent
    with downstream cosine similarity (Phase 5+).
    """

    def __init__(
        self,
        *,
        model_name: str,
        dimensions: int,
        normalize_embeddings: bool = True,
        name: str = "local_sentence_transformers",
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if dimensions <= 0:
            raise ValueError("dimensions must be > 0")
        self._model_name = model_name.strip()
        self._declared_dims = dimensions
        self._resolved_dims: int | None = None
        self._normalize_embeddings = normalize_embeddings
        self._name = name
        self._model: Any = None

        try:
            import sentence_transformers  # noqa: F401 — dependency check
        except ImportError as e:  # pragma: no cover - exercised via test when dep missing
            raise RuntimeError(_INSTALL_HINT) from e

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimensions(self) -> int:
        return self._resolved_dims if self._resolved_dims is not None else self._declared_dims

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        out_dim = int(self._model.get_sentence_embedding_dimension())
        if out_dim != self._declared_dims:
            raise ValueError(
                f"Model {self._model_name!r} outputs dimension {out_dim}, "
                f"but embedding_dimensions is {self._declared_dims} (must match for Phase 4B)"
            )
        self._resolved_dims = out_dim
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            normalize_embeddings=self._normalize_embeddings,
            convert_to_numpy=True,
        )
        return [row.tolist() for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
