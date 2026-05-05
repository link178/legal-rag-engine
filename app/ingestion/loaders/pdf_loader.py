"""PDF loader (.pdf) — text extraction only, no OCR."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.ingestion.errors import DocumentLoadError
from app.ingestion.loaders.base import LoadedDocument

_NOISE_TITLES: frozenset[str] = frozenset({"untitled", "no title"})
_NOISE_AUTHORS: frozenset[str] = frozenset({"anonymous", "unknown"})


class PdfLoader:
    supported_extensions: tuple[str, ...] = (".pdf",)

    def load(self, path: Path) -> LoadedDocument:
        try:
            reader = PdfReader(path, strict=False)
        except PdfReadError as e:
            raise DocumentLoadError(f"Failed to read PDF: {e}") from e
        except OSError as e:
            raise DocumentLoadError(f"Failed to read file: {e}") from e
        except Exception as e:
            raise DocumentLoadError(f"Failed to read PDF: {e}") from e

        if getattr(reader, "is_encrypted", False):
            try:
                decrypt_rc = reader.decrypt("")
            except Exception as e:
                raise DocumentLoadError("PDF is encrypted (password required)") from e
            if decrypt_rc == 0:
                raise DocumentLoadError("PDF is encrypted (password required)")

        page_count = len(reader.pages)

        parts: list[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception as e:
                raise DocumentLoadError(f"Failed to extract PDF text: {e}") from e
            parts.append(t.strip())

        raw_text = "\n\n".join(p for p in parts if p)
        if not raw_text.strip():
            raise DocumentLoadError(
                "PDF has no extractable text (likely scanned; OCR not supported)"
            )

        pdf_title: str | None = None
        pdf_author: str | None = None
        meta_obj = reader.metadata
        if meta_obj is not None:
            if getattr(meta_obj, "title", None):
                tt = str(meta_obj.title).strip()
                if tt and tt.lower() not in _NOISE_TITLES:
                    pdf_title = tt
            if getattr(meta_obj, "author", None):
                aa = str(meta_obj.author).strip()
                if aa and aa.lower() not in _NOISE_AUTHORS:
                    pdf_author = aa

        stem = path.stem
        title = pdf_title or (stem if stem else None)

        metadata: dict = {"format": "pdf", "page_count": page_count}
        if pdf_title is not None:
            metadata["pdf_title"] = pdf_title
        if pdf_author is not None:
            metadata["pdf_author"] = pdf_author

        return LoadedDocument(
            source_path=str(path.resolve()),
            source_type="pdf",
            raw_text=raw_text,
            title=title,
            metadata=metadata,
        )
