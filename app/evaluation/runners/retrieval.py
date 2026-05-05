"""Run retrieval evaluation over a golden JSONL file (read-only DB)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.evaluation.metrics.retrieval import (
    compute_reciprocal_rank,
    evaluate_question,
    extracted_lists_from_hits,
    summarize_retrieval_evaluation,
)
from app.evaluation.models import (
    RetrievalEvaluationItem,
    RetrievalEvaluationSummary,
    RetrievalGoldenQuestion,
)
from app.retrieval.dense import DenseRetriever
from app.retrieval.errors import ManifestNotFoundError, RetrievalError
from app.retrieval.manifest import resolve_manifest_record
from app.retrieval.models import RetrievalConfig
from app.retrieval.orchestrator import RetrievalOrchestrator
from app.retrieval.sparse import SparseRetriever
from app.storage.postgres.models import IndexManifestRecord


def _parse_uuid_list(raw: object, *, field: str) -> tuple[UUID, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a list of UUID strings")
    out: list[UUID] = []
    for i, x in enumerate(raw):
        try:
            out.append(UUID(str(x)))
        except ValueError as e:
            raise ValueError(f"{field}[{i}] is not a valid UUID: {x!r}") from e
    return tuple(out)


def _parse_str_list(raw: object, *, field: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(str(x) for x in raw)


def load_golden_questions(path: Path) -> tuple[RetrievalGoldenQuestion, ...]:
    """Load and validate golden questions from JSONL (one JSON object per non-empty line)."""
    if not path.is_file():
        raise FileNotFoundError(str(path))
    seen_ids: set[str] = set()
    out: list[RetrievalGoldenQuestion] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_no}: line must be a JSON object")
            notes = raw.get("notes")
            notes_norm = None if notes is None else str(notes)
            if notes_norm is not None and not notes_norm.strip():
                notes_norm = None
            g = RetrievalGoldenQuestion(
                id=str(raw.get("id", "")),
                question=str(raw.get("question", "")),
                expected_terms=_parse_str_list(raw.get("expected_terms"), field="expected_terms"),
                expected_document_ids=_parse_uuid_list(
                    raw.get("expected_document_ids"), field="expected_document_ids"
                ),
                expected_chunk_ids=_parse_uuid_list(
                    raw.get("expected_chunk_ids"), field="expected_chunk_ids"
                ),
                expected_source_paths=_parse_str_list(
                    raw.get("expected_source_paths"), field="expected_source_paths"
                ),
                notes=notes_norm,
            )
            if g.id in seen_ids:
                raise ValueError(f"{path}:{line_no}: duplicate question id {g.id!r}")
            seen_ids.add(g.id)
            out.append(g)
    return tuple(out)


class RetrievalEvaluationRunner:
    """
    Evaluate golden questions using a single resolved manifest and injected orchestrator.

    Resolves the index manifest once; per-question retrieval errors become items with
    ``error`` set (run continues). Does not write ``processing_runs`` or any other rows.
    """

    def __init__(
        self,
        session: Session,
        orchestrator: RetrievalOrchestrator,
        manifest_row: IndexManifestRecord,
        base_config: RetrievalConfig,
    ) -> None:
        self._session = session  # API parity; session owned by caller / retrievers
        self._orch = orchestrator
        self._manifest = manifest_row
        self._base = base_config

    @classmethod
    def from_session(
        cls,
        session: Session,
        config: RetrievalConfig,
    ) -> RetrievalEvaluationRunner:
        """Build runner with default dense/sparse retrievers and resolved manifest."""
        for_dense = config.mode != "sparse_only"
        mf = resolve_manifest_record(session, config, for_dense=for_dense)
        orch = RetrievalOrchestrator(
            DenseRetriever(session),
            SparseRetriever(session),
        )
        return cls(session, orch, mf, config)

    def run_file(self, path: Path, *, top_k: int | None = None) -> RetrievalEvaluationSummary:
        questions = load_golden_questions(path)
        return self.run_questions(questions, top_k=top_k)

    def run_questions(
        self,
        questions: Sequence[RetrievalGoldenQuestion],
        *,
        top_k: int | None = None,
    ) -> RetrievalEvaluationSummary:
        tk = top_k if top_k is not None else self._base.top_k
        cfg = RetrievalConfig(
            mode=self._base.mode,
            top_k=tk,
            dense_top_k=self._base.dense_top_k,
            sparse_top_k=self._base.sparse_top_k,
            rrf_k=self._base.rrf_k,
            index_manifest_id=self._base.index_manifest_id,
            embedding_provider=self._base.embedding_provider,
            embedding_model=self._base.embedding_model,
            embedding_dimensions=self._base.embedding_dimensions,
            chunking_strategy=self._base.chunking_strategy,
            metadata_filter=self._base.metadata_filter,
        )
        created_at = datetime.now(UTC)
        mf = self._manifest
        if cfg.mode == "hybrid" and not mf.include_sparse:
            raise ManifestNotFoundError(
                "Hybrid evaluation requires a manifest with sparse indexing "
                "(include_sparse=true). Re-run: python -m app.indexing.cli without --no-sparse"
            )
        items: list[RetrievalEvaluationItem] = []

        mf_rep = None
        if cfg.metadata_filter is not None and not cfg.metadata_filter.is_empty():
            mf_rep = dict(cfg.metadata_filter.as_dict())
        config_report = {
            "mode": cfg.mode,
            "top_k": tk,
            "rrf_k": cfg.rrf_k,
            "dense_top_k": cfg.dense_top_k,
            "sparse_top_k": cfg.sparse_top_k,
            "embedding_provider": cfg.embedding_provider,
            "embedding_model": cfg.embedding_model,
            "embedding_dimensions": cfg.embedding_dimensions,
            "chunking_strategy": cfg.chunking_strategy,
            "metadata_filter": mf_rep,
        }
        manifest_report = {
            "id": str(mf.id) if mf.id else None,
            "manifest_hash": mf.manifest_hash,
            "include_sparse": mf.include_sparse,
            "embeddings_persisted": mf.embeddings_persisted,
            "embedding_provider": mf.embedding_provider,
            "embedding_model": mf.embedding_model,
            "embedding_dimensions": mf.embedding_dimensions,
            "chunking_strategy": mf.chunking_strategy,
        }

        for g in questions:
            try:
                res = self._orch.retrieve(g.question, cfg, mf)
                hits = res.results
                hit, hit_rank, matched_by = evaluate_question(g, hits, top_k=tk)
                rr = compute_reciprocal_rank(hit_rank)
                consider = list(hits)[:tk]
                rcids, rdids, rpaths, rscores = extracted_lists_from_hits(consider)
                items.append(
                    RetrievalEvaluationItem(
                        question_id=g.id,
                        question=g.question,
                        mode=cfg.mode,
                        top_k=tk,
                        hit=hit,
                        hit_rank=hit_rank,
                        reciprocal_rank=rr,
                        matched_by=matched_by,
                        retrieved_chunk_ids=rcids,
                        retrieved_document_ids=rdids,
                        retrieved_source_paths=rpaths,
                        retrieved_scores=rscores,
                        error=None,
                    )
                )
            except RetrievalError as e:
                items.append(
                    RetrievalEvaluationItem(
                        question_id=g.id,
                        question=g.question,
                        mode=cfg.mode,
                        top_k=tk,
                        hit=False,
                        hit_rank=None,
                        reciprocal_rank=0.0,
                        matched_by=(),
                        retrieved_chunk_ids=(),
                        retrieved_document_ids=(),
                        retrieved_source_paths=(),
                        retrieved_scores=(),
                        error=str(e),
                    )
                )

        total, answered, errored, hit_rate, mrr = summarize_retrieval_evaluation(tuple(items))
        return RetrievalEvaluationSummary(
            schema_version="evaluation.retrieval.v1",
            created_at=created_at,
            config=config_report,
            manifest=manifest_report,
            total_questions=total,
            answered_questions=answered,
            errored_questions=errored,
            hit_rate=hit_rate,
            mrr=mrr,
            items=tuple(items),
        )
