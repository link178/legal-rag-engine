"""``index_chunks_persisted`` dense persistence (mocked DB layer)."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.domain.models import Chunk, ProcessingRun
from app.indexing.models import IndexingConfig
from app.indexing.runner import index_chunks_persisted
from app.indexing.service import IndexingService, default_indexing_service


def _chunk(i: int, cid=None) -> Chunk:
    cid = cid or uuid4()
    doc_id = uuid4()
    return Chunk(
        id=cid,
        document_id=doc_id,
        chunk_index=i,
        text=f"paragraph {i} legal terms",
        chunking_strategy="fixed_size",
        char_count=20,
        checksum=f"chk{i}",
    )


def _manifest_row_shell(mv_id, run_id, **overrides: object) -> MagicMock:
    m = MagicMock()
    m.id = mv_id
    m.run_id = run_id
    m.corpus_version = None
    m.chunking_strategy = "fixed_size"
    m.embedding_provider = "deterministic_hash"
    m.embedding_dimensions = 16
    m.embedding_model = None
    m.embeddings_persisted = False
    m.document_count = 1
    m.chunk_count = 2
    m.indexed_chunk_count = 0
    m.include_sparse = False
    m.include_dense = True
    m.config_hash = "cfg"
    m.chunk_set_hash = "csh"
    m.manifest_hash = "m" * 64
    m.metadata_json = {}
    m.created_at = None
    m.updated_at = None
    for k, v in overrides.items():
        setattr(m, k, v)
    return m


def test_runner_calls_embed_repo_once_per_chunk() -> None:
    c0, c1 = _chunk(0), _chunk(1)
    run_id = uuid4()
    mv_id = uuid4()

    run_repo = MagicMock()
    run_repo.add.return_value = ProcessingRun(
        run_type="indexing", status="running", id=run_id, started_at=None
    )
    run_repo.get_by_id.return_value = ProcessingRun(
        run_type="indexing", status="completed", id=run_id, started_at=None
    )

    chunk_repo = MagicMock()
    chunk_repo.list_by_chunking_strategy.return_value = [c0, c1]

    mh_row = _manifest_row_shell(mv_id, run_id, chunk_count=2, include_sparse=False)

    manifest_repo = MagicMock()
    manifest_repo.get_latest_by_manifest_hash.return_value = None
    manifest_repo.add_manifest_row.return_value = mh_row

    embed_repo = MagicMock()

    svc = default_indexing_service(dimensions=16)
    cfg = IndexingConfig(
        chunking_strategy="fixed_size",
        embedding_dimensions=16,
        batch_size=8,
        include_dense=True,
        include_sparse=False,
    )

    @contextmanager
    def _sess(_database_url=None):
        s = MagicMock()
        s.get.return_value = mh_row
        yield s

    with patch("app.indexing.runner.session_scope", _sess):
        with patch("app.indexing.runner.ProcessingRunRepository", lambda _s: run_repo):
            with patch("app.indexing.runner.ChunkRepository", lambda _s: chunk_repo):
                with patch("app.indexing.runner.IndexManifestRepository", lambda _s: manifest_repo):
                    with patch(
                        "app.indexing.runner.ChunkEmbeddingRepository",
                        lambda _s: embed_repo,
                    ):
                        res = index_chunks_persisted(
                            cfg, database_url="postgresql://x", indexing_service=svc
                        )

    assert res.error is None
    assert res.manifest is not None
    assert res.manifest.embeddings_persisted is True
    assert embed_repo.add.call_count == 2
    first_kw = embed_repo.add.call_args_list[0].kwargs
    assert first_kw["index_manifest_id"] == mv_id
    assert first_kw["embedding_dimensions"] == 16
    assert len(first_kw["embedding"]) == 16


def test_runner_skips_noop_when_manifest_embeddings_persisted() -> None:
    c0 = _chunk(0)
    run_id = uuid4()
    existing = MagicMock()
    existing.id = uuid4()
    existing.embeddings_persisted = True

    run_repo = MagicMock()
    run_repo.add.return_value = ProcessingRun(
        run_type="indexing", status="running", id=run_id, started_at=None
    )
    run_repo.get_by_id.return_value = ProcessingRun(
        run_type="indexing", status="completed", id=run_id, started_at=None
    )

    idx_repo = MagicMock()
    idx_repo.get_latest_by_manifest_hash.return_value = existing

    chunk_repo = MagicMock()
    chunk_repo.list_by_chunking_strategy.return_value = [c0]

    embed_repo = MagicMock()

    cfg = IndexingConfig(chunking_strategy="fixed_size", embedding_dimensions=16)

    @contextmanager
    def _sess(_database_url=None):
        yield MagicMock()

    with patch("app.indexing.runner.session_scope", _sess):
        with patch("app.indexing.runner.ProcessingRunRepository", lambda _s: run_repo):
            with patch("app.indexing.runner.ChunkRepository", lambda _s: chunk_repo):
                with patch("app.indexing.runner.IndexManifestRepository", lambda _s: idx_repo):
                    with patch(
                        "app.indexing.runner.ChunkEmbeddingRepository",
                        lambda _s: embed_repo,
                    ):
                        res = index_chunks_persisted(cfg, database_url="postgresql://x")

    assert res.skipped_existing is True
    embed_repo.add.assert_not_called()
    idx_repo.add_manifest_row.assert_not_called()


def test_runner_dimension_mismatch_sets_embeddings_persisted_false() -> None:
    """Provider dimension 4 vs config.embedding_dimensions 16 → no inserts, flag false."""

    class BadProvider:
        @property
        def name(self) -> str:
            return "bad"

        @property
        def dimensions(self) -> int:
            return 4

        def embed_texts(self, texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3, 0.4]

    c0 = _chunk(0)
    run_id = uuid4()
    mv_id = uuid4()

    run_repo = MagicMock()
    run_repo.add.return_value = ProcessingRun(
        run_type="indexing", status="running", id=run_id, started_at=None
    )
    run_repo.get_by_id.return_value = ProcessingRun(
        run_type="indexing", status="completed", id=run_id, started_at=None
    )

    chunk_repo = MagicMock()
    chunk_repo.list_by_chunking_strategy.return_value = [c0]

    mh_row = _manifest_row_shell(mv_id, run_id, chunk_count=1, include_sparse=False)

    manifest_repo = MagicMock()
    manifest_repo.get_latest_by_manifest_hash.return_value = None
    manifest_repo.add_manifest_row.return_value = mh_row

    embed_repo = MagicMock()
    svc = IndexingService(BadProvider())

    cfg = IndexingConfig(
        chunking_strategy="fixed_size",
        embedding_dimensions=16,
        include_sparse=False,
    )

    @contextmanager
    def _sess(_database_url=None):
        s = MagicMock()
        s.get.return_value = mh_row
        yield s

    with patch("app.indexing.runner.session_scope", _sess):
        with patch("app.indexing.runner.ProcessingRunRepository", lambda _s: run_repo):
            with patch("app.indexing.runner.ChunkRepository", lambda _s: chunk_repo):
                with patch("app.indexing.runner.IndexManifestRepository", lambda _s: manifest_repo):
                    with patch(
                        "app.indexing.runner.ChunkEmbeddingRepository",
                        lambda _s: embed_repo,
                    ):
                        res = index_chunks_persisted(
                            cfg, database_url="postgresql://x", indexing_service=svc
                        )

    assert res.error is None
    assert mh_row.embeddings_persisted is False
    embed_repo.add.assert_not_called()
    assert res.chunk_results[0].error is not None
    assert res.chunk_results[0].dense_indexed is False
