"""Operator-triggered persisted indexing run (Phase 4B: manifest + optional pgvector rows)."""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain.models import ProcessingRun
from app.indexing.dense.factory import build_embedding_provider
from app.indexing.models import (
    IndexedChunkResult,
    IndexingConfig,
    IndexingRunResult,
    IndexManifest,
    chunk_set_hash_from_fingerprints,
    fingerprint_chunk_row,
)
from app.indexing.models import (
    manifest_hash as compute_manifest_hash,
)
from app.indexing.service import IndexingService
from app.storage.postgres.models import IndexManifestRecord
from app.storage.postgres.repositories import (
    ChunkEmbeddingRepository,
    ChunkRepository,
    IndexManifestRepository,
    ProcessingRunRepository,
)
from app.storage.postgres.session import session_scope


def _manifest_row_to_domain(row: IndexManifestRecord) -> IndexManifest:
    """Map ORM row to indexing domain manifest."""
    return IndexManifest(
        id=row.id,
        run_id=row.run_id,
        corpus_version=row.corpus_version,
        chunking_strategy=row.chunking_strategy,
        embedding_provider=row.embedding_provider,
        embedding_dimensions=row.embedding_dimensions,
        embedding_model=row.embedding_model,
        embeddings_persisted=row.embeddings_persisted,
        document_count=row.document_count,
        chunk_count=row.chunk_count,
        indexed_chunk_count=row.indexed_chunk_count,
        include_sparse=row.include_sparse,
        include_dense=row.include_dense,
        config_hash=row.config_hash,
        chunk_set_hash=row.chunk_set_hash,
        manifest_hash=row.manifest_hash,
        metadata=dict(row.metadata_json),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _make_indexing_service(config: IndexingConfig) -> IndexingService:
    provider = build_embedding_provider(config)
    return IndexingService(provider)


def index_chunks_persisted(
    config: IndexingConfig,
    *,
    database_url: str | None = None,
    corpus_version: str | None = None,
    indexing_service: IndexingService | None = None,
) -> IndexingRunResult:
    """
    Persist run + manifest chunks + optional dense vectors.

    Optionally skip when an up-to-date manifest with ``embeddings_persisted`` exists.
    """
    now = datetime.now(UTC)
    run_domain = ProcessingRun(
        run_type="indexing",
        status="running",
        started_at=now,
        metadata={"source": "indexing.runner"},
    )
    try:
        with session_scope(database_url) as session:
            run_repo = ProcessingRunRepository(session)
            chunk_repo = ChunkRepository(session)
            manifest_repo = IndexManifestRepository(session)
            embed_repo = ChunkEmbeddingRepository(session)

            persisted_run = run_repo.add(run_domain)
            run_id = persisted_run.id
            assert run_id is not None

            chunks = chunk_repo.list_by_chunking_strategy(config.chunking_strategy)
            if not chunks:
                run_repo.mark_failed(
                    run_id,
                    "No chunks match the indexing filter (ingest/chunk rows first)",
                    metadata_patch={
                        "chunking_strategy_filter": config.chunking_strategy,
                    },
                )
                return IndexingRunResult(
                    manifest=None,
                    run=run_repo.get_by_id(run_id),
                    skipped_existing=False,
                    error="No chunks match the indexing filter (ingest/chunk rows first)",
                )

            fingerprints = [
                fingerprint_chunk_row(
                    c.id,
                    chunking_strategy=c.chunking_strategy,
                    checksum=c.checksum,
                )
                for c in chunks
                if c.id is not None
            ]
            cfg_hash = config.config_hash()
            cs_hash = chunk_set_hash_from_fingerprints(fingerprints)
            mh = compute_manifest_hash(cfg_hash, cs_hash)

            existing = manifest_repo.get_latest_by_manifest_hash(mh)
            if (
                existing is not None
                and existing.embeddings_persisted
                and not config.force_reindex
            ):
                run_repo.mark_completed(
                    run_id,
                    chunks_created=len(chunks),
                    metadata_patch={
                        "skipped": True,
                        "manifest_hash": mh,
                        "existing_manifest_id": str(existing.id),
                    },
                )
                return IndexingRunResult(
                    manifest=_manifest_row_to_domain(existing),
                    run=run_repo.get_by_id(run_id),
                    skipped_existing=True,
                )

            svc: IndexingService
            if indexing_service is not None:
                svc = indexing_service
            else:
                svc = _make_indexing_service(config)

            indexed_ok = 0
            chunk_results: list[IndexedChunkResult] = []
            embeddings_inserted = 0

            document_count = len({c.document_id for c in chunks})
            embedding_model_val = config.normalized_embedding_model()
            mh_row = manifest_repo.add_manifest_row(
                run_id=run_id,
                corpus_version=corpus_version,
                chunking_strategy=config.chunking_strategy,
                embedding_provider=config.embedding_provider.strip(),
                embedding_dimensions=config.embedding_dimensions,
                embedding_model=embedding_model_val,
                embeddings_persisted=False,
                document_count=document_count,
                chunk_count=len(chunks),
                indexed_chunk_count=0,
                include_sparse=config.include_sparse,
                include_dense=config.include_dense,
                config_hash=cfg_hash,
                chunk_set_hash=cs_hash,
                manifest_hash=mh,
                metadata_json={"phase": "4b", "skipped": False},
            )
            mv_row_id = mh_row.id

            batch_size = config.batch_size
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                perf_results, dense_out, sparse_maps = svc.index_chunks(batch, config)
                for i, c in enumerate(batch):
                    r_prior = perf_results[i]
                    sparse_dict = sparse_maps[i] if sparse_maps is not None else None

                    err = r_prior.error
                    dense_flag = r_prior.dense_indexed
                    sparse_flag = r_prior.sparse_indexed
                    vec: list[float] | None = None

                    if config.include_dense:
                        if dense_out is None:
                            err = err or "missing dense output batch"
                            dense_flag = False
                        else:
                            vec = dense_out[i]
                            if len(vec) != config.embedding_dimensions:
                                err = (
                                    f"dimension mismatch: got {len(vec)}, "
                                    f"expected {config.embedding_dimensions}"
                                )
                                dense_flag = False
                                vec = None

                    r = IndexedChunkResult(
                        chunk_id=r_prior.chunk_id,
                        dense_indexed=dense_flag,
                        sparse_indexed=sparse_flag,
                        error=err,
                    )

                    manifest_repo.add_manifest_chunk_row(
                        manifest_id=mv_row_id,
                        chunk_id=r.chunk_id,
                        dense_indexed=r.dense_indexed,
                        sparse_indexed=r.sparse_indexed,
                        sparse_terms=sparse_dict if config.include_sparse else None,
                        error_message=r.error,
                    )
                    chunk_results.append(r)

                    if (
                        config.include_dense
                        and r.dense_indexed
                        and r.error is None
                        and vec is not None
                    ):
                        embed_repo.add(
                            chunk_id=r.chunk_id,
                            index_manifest_id=mv_row_id,
                            embedding_provider=config.embedding_provider.strip(),
                            embedding_model=embedding_model_val,
                            embedding_dimensions=config.embedding_dimensions,
                            embedding=vec,
                            text_checksum=c.checksum or "",
                        )
                        embeddings_inserted += 1

                    if r.error is None:
                        indexed_ok += 1

            mh_row.indexed_chunk_count = indexed_ok
            mh_row.document_count = document_count

            if config.include_dense:
                mh_row.embeddings_persisted = all(
                    x.error is None and x.dense_indexed for x in chunk_results
                )
            else:
                mh_row.embeddings_persisted = False

            session.flush()

            run_repo.mark_completed(
                run_id,
                chunks_created=len(chunk_results),
                metadata_patch={
                    "manifest_hash": mh,
                    "manifest_id": str(mv_row_id),
                    "indexed_chunk_count": indexed_ok,
                    "embeddings_inserted": embeddings_inserted,
                },
            )

            refreshed = session.get(IndexManifestRecord, mv_row_id)
            dom = _manifest_row_to_domain(refreshed) if refreshed else None
            return IndexingRunResult(
                manifest=dom,
                run=run_repo.get_by_id(run_id),
                skipped_existing=False,
                chunk_results=chunk_results,
            )
    except Exception as e:  # noqa: BLE001 — DB connectivity etc.; transaction may roll back
        return IndexingRunResult(
            manifest=None,
            run=None,
            skipped_existing=False,
            error=f"{type(e).__name__}: {e}",
        )
