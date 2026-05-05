"""Local legal corpus adapter (e.g. legalize-* layout) for operator-driven imports."""

from app.ingestion.adapters.legal_corpus.cli import main as cli_main
from app.ingestion.adapters.legal_corpus.discovery import scan_corpus
from app.ingestion.adapters.legal_corpus.importer import (
    DefaultCorpusRunRecorder,
    LegalCorpusImporter,
)
from app.ingestion.adapters.legal_corpus.models import (
    LegalCorpusDocumentCandidate,
    LegalCorpusImportItem,
    LegalCorpusImportSummary,
)

__all__ = [
    "cli_main",
    "DefaultCorpusRunRecorder",
    "LegalCorpusDocumentCandidate",
    "LegalCorpusImportItem",
    "LegalCorpusImportSummary",
    "LegalCorpusImporter",
    "scan_corpus",
]
