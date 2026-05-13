"""Fixed-size chunking with overlap and simple boundary preference."""

from __future__ import annotations

from uuid import UUID

from app.chunking.errors import DocumentIdRequiredError, EmptyDocumentError
from app.chunking.helpers import chunk_checksum, estimate_tokens, merge_chunk_metadata
from app.chunking.models import ChunkingConfig
from app.domain.models import Chunk, Document

STRATEGY_NAME = "fixed_size"


def _adjust_boundary_end(text: str, start: int, max_end: int) -> int:
    """Prefer break near end of window: ``\\n\\n``, ``\\n``, space, else hard ``max_end``."""
    if max_end >= len(text):
        return len(text)
    segment = text[start:max_end]
    for sep in ("\n\n", "\n", " "):
        idx = segment.rfind(sep)
        if idx != -1:
            cut = start + idx + len(sep)
            if cut > start:
                return cut
    return max_end


def split_text_fixed(
    text: str,
    *,
    document_id: UUID,
    document_checksum: str,
    config: ChunkingConfig,
    cfg_hash: str,
    cfg_normalized: dict[str, int | bool | str],
    strategy_label: str,
    heading: str | None = None,
    section_index: int | None = None,
    base_offset: int = 0,
) -> list[Chunk]:
    """Split ``text`` into fixed-size windows; absolute offsets use ``base_offset``."""
    n = len(text)
    chunks: list[Chunk] = []
    chunk_index_local = 0
    start = 0

    while start < n:
        if n - start <= config.chunk_size:
            end = n
        else:
            tentative_max = min(start + config.chunk_size, n)
            end = _adjust_boundary_end(text, start, tentative_max)
            if end <= start:
                end = tentative_max

            segment = text[start:end]
            if end < n and len(segment) < config.min_chunk_chars:
                stretch = min(max(start + config.min_chunk_chars, end), n)
                end = _adjust_boundary_end(text, start, stretch)
                if end <= start:
                    end = min(start + config.chunk_size, n)

        segment = text[start:end]
        if not segment.strip():
            start = start + 1
            continue

        char_count = len(segment)
        meta_extra: dict = {}
        if heading is not None:
            meta_extra["heading"] = heading
        if section_index is not None:
            meta_extra["section_index"] = section_index

        md = merge_chunk_metadata(
            {},
            start_offset=base_offset + start,
            end_offset=base_offset + end,
            chunking_config_hash=cfg_hash,
            chunking_config=dict(cfg_normalized),
            source_document_checksum=document_checksum,
            extra=meta_extra if meta_extra else None,
        )

        chunks.append(
            Chunk(
                document_id=document_id,
                chunk_index=chunk_index_local,
                text=segment,
                chunking_strategy=strategy_label,
                char_count=char_count,
                heading=heading,
                token_estimate=estimate_tokens(char_count),
                metadata=md,
                checksum=chunk_checksum(segment),
            )
        )
        chunk_index_local += 1

        if end >= n:
            break

        next_start = end - config.chunk_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


class FixedSizeChunkingStrategy:
    """Character windows with overlap; prefers newline/space boundaries."""

    name = STRATEGY_NAME

    def split(self, document: Document, config: ChunkingConfig) -> list[Chunk]:
        if document.id is None:
            raise DocumentIdRequiredError(
                "Chunking requires a persisted document id; ingest with --persist first."
            )
        text = document.normalized_text or document.raw_text or ""
        if not text.strip():
            raise EmptyDocumentError("Document has no normalized or raw text to chunk.")

        cfg = ChunkingConfig(
            strategy=config.strategy,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_chars=config.min_chunk_chars,
            preserve_headings=config.preserve_headings,
        )
        cfg_hash = cfg.config_hash()
        cfg_norm = cfg.normalized_dict()

        raw = split_text_fixed(
            text,
            document_id=document.id,
            document_checksum=document.checksum,
            config=cfg,
            cfg_hash=cfg_hash,
            cfg_normalized=cfg_norm,
            strategy_label=self.name,
            base_offset=0,
        )
        return [_with_chunk_index(ch, i) for i, ch in enumerate(raw)]


def _with_chunk_index(ch: Chunk, index: int) -> Chunk:
    return Chunk(
        document_id=ch.document_id,
        chunk_index=index,
        text=ch.text,
        chunking_strategy=ch.chunking_strategy,
        char_count=ch.char_count,
        id=ch.id,
        created_by_run_id=ch.created_by_run_id,
        heading=ch.heading,
        page_number=ch.page_number,
        token_estimate=ch.token_estimate,
        metadata=dict(ch.metadata),
        checksum=ch.checksum,
        created_at=ch.created_at,
        updated_at=ch.updated_at,
    )
