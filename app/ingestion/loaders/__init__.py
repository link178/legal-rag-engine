"""Document loaders."""

from app.ingestion.loaders.base import DocumentLoader, LoadedDocument
from app.ingestion.loaders.markdown_loader import MarkdownLoader
from app.ingestion.loaders.text_loader import TextLoader

__all__ = [
    "DocumentLoader",
    "LoadedDocument",
    "MarkdownLoader",
    "TextLoader",
]
