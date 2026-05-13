"""Document loaders."""

from app.ingestion.loaders.base import DocumentLoader, LoadedDocument
from app.ingestion.loaders.html_loader import HtmlLoader
from app.ingestion.loaders.markdown_loader import MarkdownLoader
from app.ingestion.loaders.pdf_loader import PdfLoader
from app.ingestion.loaders.text_loader import TextLoader

__all__ = [
    "DocumentLoader",
    "HtmlLoader",
    "LoadedDocument",
    "MarkdownLoader",
    "PdfLoader",
    "TextLoader",
]
