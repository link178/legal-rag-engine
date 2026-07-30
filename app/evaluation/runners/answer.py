"""Run answer evaluation over a golden JSONL file (read-only DB)."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.evaluation.metrics.answer import evaluate_answer_question, summarize_answer_evaluation
from app.evaluation.models import (
    AnswerEvaluationItem,
    AnswerEvaluationSummary,
    AnswerGoldenQuestion,
)
from app.evaluation.status import derive_answer_execution_status
from app.generation.answerer import GroundedAnswerer
from app.generation.context import ContextBuilder
from app.generation.providers import MockGenerationProvider
from app.generation.providers.base import GenerationProvider
from app.retrieval.errors import ManifestNotFoundError, RetrievalError
from app.retrieval.manifest import resolve_manifest_record
from app.retrieval.models import RetrievalConfig
from app.storage.postgres.models import IndexManifestRecord


def _parse_str_list(raw: object, *, field: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(str(x) for x in raw)


def _parse_optional_str(raw: object, *, field: str) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def _parse_optional_notes(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw.strip() or None
    return str(raw).strip() or None


def _parse_bool(raw: object, *, field: str, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(raw)
    raise ValueError(f"{field} must be a boolean")


def load_answer_golden(path: Path) -> tuple[AnswerGoldenQuestion, ...]:
    """Load answer golden questions from JSONL (one JSON object per non-empty line)."""
    if not path.is_file():
        raise FileNotFoundError(str(path))
    seen_ids: set[str] = set()
    out: list[AnswerGoldenQuestion] = []
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
            eti_raw = _parse_optional_str(raw.get("expected_terms_in"), field="expected_terms_in")
            g = AnswerGoldenQuestion(
                id=str(raw.get("id", "")),
                question=str(raw.get("question", "")),
                expected_mode=_parse_optional_str(raw.get("expected_mode"), field="expected_mode"),
                expected_terms=_parse_str_list(raw.get("expected_terms"), field="expected_terms"),
                expected_terms_in=eti_raw or "both",
                expected_source_paths=_parse_str_list(
                    raw.get("expected_source_paths"),
                    field="expected_source_paths",
                ),
                should_be_insufficient_context=_parse_bool(
                    raw.get("should_be_insufficient_context"),
                    field="should_be_insufficient_context",
                    default=False,
                ),
                recommended_mode=_parse_optional_str(
                    raw.get("recommended_mode"), field="recommended_mode"
                ),
                notes=_parse_optional_notes(raw.get("notes")),
            )
            if g.id in seen_ids:
                raise ValueError(f"{path}:{line_no}: duplicate question id {g.id!r}")
            seen_ids.add(g.id)
            out.append(g)
    return tuple(out)


def _error_item(
    g: AnswerGoldenQuestion,
    *,
    retrieval_mode: str,
    exc: Exception,
) -> AnswerEvaluationItem:
    return AnswerEvaluationItem(
        question_id=g.id,
        question=g.question,
        mode=retrieval_mode,
        answer_mode="error",
        expected_mode=g.expected_mode,
        mode_matches=False,
        contains_expected_terms=False,
        missing_expected_terms=(),
        citation_validity_rate=0.0,
        has_valid_citations=False,
        has_invalid_citations=False,
        insufficient_context_matches=False,
        retrieved_expected_source=None,
        cited_source_paths=(),
        used_citation_ids=(),
        invalid_citation_ids=(),
        passed=False,
        error=str(exc),
    )


def _snapshot_context(cb: ContextBuilder) -> dict[str, object]:
    """Report ContextBuilder limits (attributes are internal; stable for Phase 11)."""
    return {
        "max_chunks": cb._max_chunks,
        "max_context_chars": cb._max_context_chars,
        "max_chunk_chars": cb._max_chunk_chars,
        "min_score": cb._min_score,
    }


class AnswerEvaluationRunner:
    """Evaluate answer goldens via ``GroundedAnswerer``. Read-only DB."""

    def __init__(
        self,
        answerer: GroundedAnswerer,
        manifest_row: IndexManifestRecord,
        base_config: RetrievalConfig,
        *,
        context_builder: ContextBuilder,
        generation_provider_name: str,
    ) -> None:
        self._answerer = answerer
        self._manifest = manifest_row
        self._base = base_config
        self._context_builder = context_builder
        self._generation_provider_name = generation_provider_name

    @classmethod
    def from_session(
        cls,
        session: Session,
        config: RetrievalConfig,
        *,
        context_builder: ContextBuilder | None = None,
        provider: GenerationProvider | None = None,
    ) -> AnswerEvaluationRunner:
        cb = context_builder or ContextBuilder()
        prov = provider if provider is not None else MockGenerationProvider()
        for_dense = config.mode != "sparse_only"
        mf = resolve_manifest_record(session, config, for_dense=for_dense)
        if config.mode == "hybrid" and not mf.include_sparse:
            raise ManifestNotFoundError(
                "Hybrid answer evaluation requires a manifest with sparse indexing "
                "(include_sparse=true). Re-run: python -m app.indexing.cli without --no-sparse"
            )
        ga = GroundedAnswerer.from_session(session, config, context_builder=cb, provider=prov)
        return cls(
            ga,
            mf,
            config,
            context_builder=cb,
            generation_provider_name=getattr(prov, "name", type(prov).__name__),
        )

    def run_file(self, path: Path, *, top_k: int | None = None) -> AnswerEvaluationSummary:
        questions = load_answer_golden(path)
        return self.run_questions(questions, top_k=top_k)

    def run_questions(
        self,
        questions: Sequence[AnswerGoldenQuestion],
        *,
        top_k: int | None = None,
    ) -> AnswerEvaluationSummary:
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
        mf = self._manifest
        if cfg.mode == "hybrid" and not mf.include_sparse:
            raise ManifestNotFoundError(
                "Hybrid answer evaluation requires a manifest with sparse indexing "
                "(include_sparse=true)."
            )

        created_at = datetime.now(UTC)
        cfg_snapshot = _snapshot_context(self._context_builder)
        mf_rep = None
        if cfg.metadata_filter is not None and not cfg.metadata_filter.is_empty():
            mf_rep = dict(cfg.metadata_filter.as_dict())
        config_report: dict[str, object] = {
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
            "provider": self._generation_provider_name,
            **cfg_snapshot,
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

        items: list[AnswerEvaluationItem] = []
        for g in questions:
            try:
                if g.recommended_mode is not None and g.recommended_mode != cfg.mode:
                    print(
                        f"warn: golden {g.id!r} recommends retrieval mode "
                        f"{g.recommended_mode!r}; running {cfg.mode!r}",
                        file=sys.stderr,
                    )
                ans = self._answerer.answer(g.question)
                items.append(
                    evaluate_answer_question(g, ans, retrieval_mode=cfg.mode),
                )
            except RetrievalError as e:
                items.append(_error_item(g, retrieval_mode=cfg.mode, exc=e))
            except ValueError as e:
                items.append(_error_item(g, retrieval_mode=cfg.mode, exc=e))

        tup = tuple(items)
        gt_has_terms = tuple(
            bool(gg.expected_terms and any(t.strip() for t in gg.expected_terms))
            for gg in questions
        )
        total, answered, errored, pr, ma, ea, cra, isa, rsa = summarize_answer_evaluation(
            tup,
            golden_has_terms=gt_has_terms,
        )
        status = derive_answer_execution_status(tup, total_questions=total)
        return AnswerEvaluationSummary(
            schema_version="evaluation.answer.v1",
            created_at=created_at,
            config=config_report,
            manifest=manifest_report,
            total_questions=total,
            answered_questions=answered,
            errored_questions=errored,
            pass_rate=pr,
            mode_accuracy=ma,
            expected_terms_accuracy=ea,
            citation_validity_rate_avg=cra,
            insufficient_context_accuracy=isa,
            retrieved_expected_source_rate=rsa,
            items=tup,
            execution_status=status,
        )
