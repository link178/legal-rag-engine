"""Orchestrate retrieval → context → prompt → generation → GroundedAnswer."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.generation.citations import extract_cited_ids
from app.generation.context import ContextBuilder
from app.generation.models import GroundedAnswer, citation_from_block
from app.generation.prompts import build_grounded_prompt
from app.generation.providers import (
    INSUFFICIENT_CONTEXT_SENTENCE,
    GenerationProvider,
    MockGenerationProvider,
)
from app.retrieval.dense import DenseRetriever
from app.retrieval.errors import EmptyQueryError, ManifestNotFoundError
from app.retrieval.manifest import resolve_manifest_record
from app.retrieval.models import RetrievalConfig
from app.retrieval.orchestrator import RetrievalOrchestrator
from app.retrieval.sparse import SparseRetriever
from app.storage.postgres.models import IndexManifestRecord


class GroundedAnswerer:
    """End-to-end grounded answer pipeline for one resolved manifest + config."""

    def __init__(
        self,
        *,
        orchestrator: RetrievalOrchestrator,
        manifest: IndexManifestRecord,
        config: RetrievalConfig,
        context_builder: ContextBuilder,
        provider: GenerationProvider,
    ) -> None:
        self._orch = orchestrator
        self._manifest = manifest
        self._config = config
        self._builder = context_builder
        self._provider = provider

    @classmethod
    def from_session(
        cls,
        session: Session,
        config: RetrievalConfig,
        *,
        context_builder: ContextBuilder | None = None,
        provider: GenerationProvider | None = None,
    ) -> GroundedAnswerer:
        for_dense = config.mode != "sparse_only"
        manifest = resolve_manifest_record(session, config, for_dense=for_dense)
        if config.mode == "hybrid" and not manifest.include_sparse:
            raise ManifestNotFoundError(
                "Hybrid generation needs a manifest built with sparse indexing "
                "(include_sparse=true). Re-run: python -m app.indexing.cli without --no-sparse"
            )
        orch = RetrievalOrchestrator(
            DenseRetriever(session),
            SparseRetriever(session),
        )
        return cls(
            orchestrator=orch,
            manifest=manifest,
            config=config,
            context_builder=context_builder or ContextBuilder(),
            provider=provider or MockGenerationProvider(),
        )

    def answer(self, question: str) -> GroundedAnswer:
        if not question.strip():
            raise EmptyQueryError("query must be non-empty")

        sentinel = INSUFFICIENT_CONTEXT_SENTENCE.strip()
        retrieval = self._orch.retrieve(question, self._config, self._manifest)
        n_retrieved = len(retrieval.results)
        mf = self._manifest
        mf_id = str(mf.id) if mf.id else None
        base_meta = {
            "generation_provider": self._provider.name,
            "retrieval_mode": retrieval.mode,
            "index_manifest_id": mf_id,
            "manifest_hash": mf.manifest_hash,
            "embedding_provider": retrieval.embedding_provider,
            "embedding_model": retrieval.embedding_model,
            "embedding_dimensions": retrieval.embedding_dimensions,
            "total_retrieved_chunks": n_retrieved,
            "total_context_blocks": 0,
            "prompt_chars": 0,
            "answer_chars": 0,
        }

        blocks = self._builder.build(retrieval.results)
        base_meta["total_context_blocks"] = len(blocks)

        if not blocks:
            ans = sentinel
            return GroundedAnswer(
                question=question,
                answer=ans,
                mode="insufficient_context",
                citations=(),
                used_citation_ids=(),
                retrieval_mode=retrieval.mode,
                insufficient_context=True,
                metadata=base_meta | {"prompt_chars": 0, "answer_chars": len(ans)},
            )

        prompt = build_grounded_prompt(
            question,
            blocks,
            insufficient_sentence=INSUFFICIENT_CONTEXT_SENTENCE,
        )
        base_meta["prompt_chars"] = len(prompt)

        raw = self._provider.generate(prompt).strip()
        base_meta["answer_chars"] = len(raw)

        if raw == sentinel:
            return GroundedAnswer(
                question=question,
                answer=INSUFFICIENT_CONTEXT_SENTENCE,
                mode="insufficient_context",
                citations=(),
                used_citation_ids=(),
                retrieval_mode=retrieval.mode,
                insufficient_context=True,
                metadata={**base_meta, "prompt_chars": len(prompt)},
            )

        ordered = extract_cited_ids(raw)
        block_ids = {b.citation_id for b in blocks}
        valid = tuple(cid for cid in ordered if cid in block_ids)
        citations = tuple(citation_from_block(b) for b in blocks)
        mode = "grounded" if valid else "partial"

        return GroundedAnswer(
            question=question,
            answer=raw,
            mode=mode,
            citations=citations,
            used_citation_ids=valid,
            retrieval_mode=retrieval.mode,
            insufficient_context=False,
            metadata=base_meta,
        )
