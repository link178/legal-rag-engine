"""Markdown heading-aware chunking with fallback to fixed-size windows."""

from __future__ import annotations

import re
from uuid import UUID

from app.chunking.errors import DocumentIdRequiredError, EmptyDocumentError
from app.chunking.models import ChunkingConfig
from app.chunking.strategies.fixed_size import STRATEGY_NAME as FIXED_NAME
from app.chunking.strategies.fixed_size import split_text_fixed
from app.domain.models import Chunk, Document

STRATEGY_NAME = "structure_aware"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class StructureAwareChunkingStrategy:
    """Splits on Markdown ATX headings when present; otherwise applies fixed-size."""

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

        matches = list(_HEADING_RE.finditer(text))
        if not matches or not cfg.preserve_headings:
            return self._fallback_fixed(
                document.id, document.checksum, text, cfg, cfg_hash, cfg_norm
            )

        chunks: list[Chunk] = []
        preamble = text[: matches[0].start()]
        sec_idx = 0
        if preamble.strip():
            sub = split_text_fixed(
                preamble,
                document_id=document.id,
                document_checksum=document.checksum,
                config=cfg,
                cfg_hash=cfg_hash,
                cfg_normalized=cfg_norm,
                strategy_label=self.name,
                heading=None,
                section_index=sec_idx,
                base_offset=0,
            )
            chunks.extend(sub)
            sec_idx += 1

        for i, m in enumerate(matches):
            level = len(m.group(1))
            heading_title = m.group(2).strip()
            line_end = text.find("\n", m.start())
            body_start = len(text) if line_end == -1 else line_end + 1
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end]

            meta_heading = heading_title
            sub = split_text_fixed(
                body,
                document_id=document.id,
                document_checksum=document.checksum,
                config=cfg,
                cfg_hash=cfg_hash,
                cfg_normalized=cfg_norm,
                strategy_label=self.name,
                heading=meta_heading,
                section_index=sec_idx,
                base_offset=body_start,
            )
            # annotate heading_level on metadata for traceability
            for ch in sub:
                md = dict(ch.metadata)
                md["heading_level"] = level
                md["source_strategy"] = self.name
                chunks.append(
                    Chunk(
                        document_id=ch.document_id,
                        chunk_index=ch.chunk_index,
                        text=ch.text,
                        chunking_strategy=ch.chunking_strategy,
                        char_count=ch.char_count,
                        heading=ch.heading,
                        token_estimate=ch.token_estimate,
                        metadata=md,
                        checksum=ch.checksum,
                    )
                )
            sec_idx += 1

        return [_with_chunk_index(ch, i) for i, ch in enumerate(chunks)]

    def _fallback_fixed(
        self,
        doc_id: UUID,
        checksum: str,
        text: str,
        cfg: ChunkingConfig,
        cfg_hash: str,
        cfg_norm: dict[str, int | bool | str],
    ) -> list[Chunk]:
        raw = split_text_fixed(
            text,
            document_id=doc_id,
            document_checksum=checksum,
            config=cfg,
            cfg_hash=cfg_hash,
            cfg_normalized=cfg_norm,
            strategy_label=self.name,
            base_offset=0,
        )
        out: list[Chunk] = []
        for ch in raw:
            md = dict(ch.metadata)
            md["source_strategy"] = FIXED_NAME
            md["fallback"] = "no_markdown_headings"
            out.append(
                Chunk(
                    document_id=ch.document_id,
                    chunk_index=ch.chunk_index,
                    text=ch.text,
                    chunking_strategy=ch.chunking_strategy,
                    char_count=ch.char_count,
                    heading=ch.heading,
                    token_estimate=ch.token_estimate,
                    metadata=md,
                    checksum=ch.checksum,
                )
            )
        return [_with_chunk_index(ch, i) for i, ch in enumerate(out)]


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
