"""Map domain exceptions to HTTP JSON envelope."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from app.api.schemas.common import ApiErrorBody, ApiErrorResponse
from app.generation.errors import GenerationError, UnsupportedProviderError
from app.ingestion.errors import (
    DocumentLoadError,
    UnsupportedDocumentTypeError,
)
from app.ingestion.errors import (
    DocumentNotFoundError as IngestDocumentNotFoundError,
)
from app.retrieval.errors import (
    EmbeddingDimensionMismatchError,
    EmptyQueryError,
    ManifestNotFoundError,
    RetrievalError,
    RetrieverNotConfiguredError,
)

log = logging.getLogger(__name__)


class IngestionFailedError(Exception):
    """Ingest runner returned result.error (persist path)."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class PersistedDocumentNotFoundError(Exception):
    """No row for GET /v1/documents/{id}."""

    pass


class ChunkingFailedError(Exception):
    """Chunk runner returned result.error (persist path)."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ChunkConfigConflictError(Exception):
    """Existing chunks for document/strategy disagree with requested config hash."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class IndexingFailedError(Exception):
    """Index runner returned result.error (persist path)."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class NoChunksToIndexError(Exception):
    """No chunk rows match indexing filter (pipeline state)."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def _json(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    body = ApiErrorResponse(
        error=ApiErrorBody(
            code=code,
            message=message,
            details=details or {},
        )
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _validation_error_details(exc: RequestValidationError) -> list[Any]:
    """JSON-safe error payload (drops non-serializable exception objects under ctx)."""

    return json.loads(json.dumps(exc.errors(), default=str))


def _pydantic_validation_error_details(exc: PydanticValidationError) -> list[Any]:
    return json.loads(exc.json(include_url=False))


def register_exception_handlers(app: FastAPI) -> None:
    """Install handlers for stable { error: { code, message, details } } responses."""

    @app.exception_handler(EmptyQueryError)
    async def _empty_query(_request: Request, exc: EmptyQueryError) -> JSONResponse:
        return _json(400, "empty_query", str(exc))

    @app.exception_handler(RetrieverNotConfiguredError)
    async def _retriever_not_configured(
        _request: Request, exc: RetrieverNotConfiguredError
    ) -> JSONResponse:
        return _json(400, "retriever_not_configured", str(exc))

    @app.exception_handler(EmbeddingDimensionMismatchError)
    async def _embedding_mismatch(
        _request: Request, exc: EmbeddingDimensionMismatchError
    ) -> JSONResponse:
        return _json(400, "embedding_dimension_mismatch", str(exc))

    @app.exception_handler(ManifestNotFoundError)
    async def _manifest_not_found(_request: Request, exc: ManifestNotFoundError) -> JSONResponse:
        return _json(404, "manifest_not_found", str(exc))

    @app.exception_handler(IngestDocumentNotFoundError)
    async def _ingest_doc_not_found(
        _request: Request, exc: IngestDocumentNotFoundError
    ) -> JSONResponse:
        return _json(404, "document_not_found", str(exc))

    @app.exception_handler(PersistedDocumentNotFoundError)
    async def _persisted_doc_not_found(
        _request: Request, _exc: PersistedDocumentNotFoundError
    ) -> JSONResponse:
        return _json(404, "document_not_found", "Document not found.")

    @app.exception_handler(UnsupportedDocumentTypeError)
    async def _unsupported_type(
        _request: Request, exc: UnsupportedDocumentTypeError
    ) -> JSONResponse:
        return _json(415, "unsupported_document_type", str(exc))

    @app.exception_handler(DocumentLoadError)
    async def _document_load(_request: Request, exc: DocumentLoadError) -> JSONResponse:
        return _json(422, "document_load_failed", str(exc))

    @app.exception_handler(UnsupportedProviderError)
    async def _unsupported_provider(
        _request: Request, exc: UnsupportedProviderError
    ) -> JSONResponse:
        return _json(400, "unsupported_provider", str(exc))

    @app.exception_handler(IngestionFailedError)
    async def _ingestion_failed(_request: Request, exc: IngestionFailedError) -> JSONResponse:
        return _json(
            422,
            "ingestion_failed",
            "Ingestion failed.",
            details={"error": exc.message},
        )

    @app.exception_handler(ChunkingFailedError)
    async def _chunking_failed(_request: Request, exc: ChunkingFailedError) -> JSONResponse:
        return _json(
            422,
            "chunking_failed",
            "Chunking failed.",
            details={"error": exc.message},
        )

    @app.exception_handler(ChunkConfigConflictError)
    async def _chunk_config_conflict(
        _request: Request, exc: ChunkConfigConflictError
    ) -> JSONResponse:
        return _json(
            409,
            "chunking_config_conflict",
            "Chunking configuration conflict.",
            details={"error": exc.message},
        )

    @app.exception_handler(IndexingFailedError)
    async def _indexing_failed(_request: Request, exc: IndexingFailedError) -> JSONResponse:
        return _json(
            422,
            "indexing_failed",
            "Indexing failed.",
            details={"error": exc.message},
        )

    @app.exception_handler(NoChunksToIndexError)
    async def _no_chunks_to_index(
        _request: Request, exc: NoChunksToIndexError
    ) -> JSONResponse:
        return _json(
            400,
            "no_chunks_to_index",
            "No chunks to index for the given filters.",
            details={"error": exc.message},
        )

    @app.exception_handler(GenerationError)
    async def _generation_error(_request: Request, exc: GenerationError) -> JSONResponse:
        return _json(500, "generation_error", str(exc))

    @app.exception_handler(RetrievalError)
    async def _retrieval_error(_request: Request, exc: RetrievalError) -> JSONResponse:
        # Subclasses registered above take precedence; this is fallback
        return _json(400, "retrieval_error", str(exc))
    @app.exception_handler(PydanticValidationError)
    async def _pydantic_validation_error(
        _request: Request,
        exc: PydanticValidationError,
    ) -> JSONResponse:
        return _json(
            422,
            "validation_error",
            "Request validation failed.",
            details={"errors": _pydantic_validation_error_details(exc)},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return _json(
            422,
            "validation_error",
            "Request validation failed.",
            details={"errors": _validation_error_details(exc)},
        )
    @app.exception_handler(ValueError)
    async def _value_error(_request: Request, exc: ValueError) -> JSONResponse:
        if isinstance(exc, PydanticValidationError):
            return _json(
                422,
                "validation_error",
                "Request validation failed.",
                details={"errors": _pydantic_validation_error_details(exc)},
            )
        return _json(400, "invalid_config", str(exc))

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error: %s", exc)
        return _json(
            500,
            "internal_error",
            "An unexpected error occurred.",
        )
