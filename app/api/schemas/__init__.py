"""API request/response schemas (Pydantic v2)."""

from __future__ import annotations

from app.api.schemas.answer import (
    AnswerRequest,
    AnswerResponse,
    CitationVerificationResponse,
    GroundedCitationResponse,
)
from app.api.schemas.chunk import (
    ChunkItemResponse,
    ChunkRequest,
    ChunkResponse,
    DocumentChunksResponse,
)
from app.api.schemas.common import ApiErrorBody, ApiErrorResponse
from app.api.schemas.documents import (
    DocumentDetailResponse,
    DocumentListItem,
    DocumentListResponse,
)
from app.api.schemas.index import (
    IndexManifestDetailResponse,
    IndexManifestItemResponse,
    IndexManifestListResponse,
    IndexRequest,
    IndexResponse,
)
from app.api.schemas.ingest import IngestRequest, IngestResponse
from app.api.schemas.retrieve import (
    RetrievalParams,
    RetrievedChunkResponse,
    RetrieveRequest,
    RetrieveResponse,
    text_preview,
)

__all__ = [
    "AnswerRequest",
    "AnswerResponse",
    "ApiErrorBody",
    "ApiErrorResponse",
    "ChunkItemResponse",
    "ChunkRequest",
    "ChunkResponse",
    "CitationVerificationResponse",
    "DocumentChunksResponse",
    "DocumentDetailResponse",
    "DocumentListItem",
    "DocumentListResponse",
    "GroundedCitationResponse",
    "IndexManifestDetailResponse",
    "IndexManifestItemResponse",
    "IndexManifestListResponse",
    "IndexRequest",
    "IndexResponse",
    "IngestRequest",
    "IngestResponse",
    "RetrievedChunkResponse",
    "RetrieveRequest",
    "RetrieveResponse",
    "RetrievalParams",
    "text_preview",
]
