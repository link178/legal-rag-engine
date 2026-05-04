"""POST /v1/retrieve — read-only; thin wrapper over manifest + orchestrator."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies.db import get_session
from app.api.mappers import retrieval_result_to_response
from app.api.retrieval_config import retrieval_config_from_params
from app.api.schemas.retrieve import RetrieveRequest, RetrieveResponse
from app.core.config import Settings, get_settings
from app.retrieval.dense import DenseRetriever
from app.retrieval.manifest import resolve_manifest_record
from app.retrieval.orchestrator import RetrievalOrchestrator
from app.retrieval.sparse import SparseRetriever

router = APIRouter()


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_chunks(
    body: RetrieveRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> RetrieveResponse:
    cfg = retrieval_config_from_params(body, settings)

    if cfg.mode == "sparse_only":
        mf = resolve_manifest_record(session, cfg, for_dense=False)
    else:
        mf = resolve_manifest_record(session, cfg, for_dense=True)
    if cfg.mode == "hybrid" and not mf.include_sparse:
        from app.retrieval.errors import ManifestNotFoundError

        raise ManifestNotFoundError(
            "Hybrid retrieval needs a manifest built with sparse indexing "
            "(include_sparse=true). Re-run: python -m app.indexing.cli without --no-sparse"
        )

    orch = RetrievalOrchestrator(
        DenseRetriever(session),
        SparseRetriever(session),
    )
    res = orch.retrieve(body.query, cfg, mf)
    return retrieval_result_to_response(res, cfg.top_k)
