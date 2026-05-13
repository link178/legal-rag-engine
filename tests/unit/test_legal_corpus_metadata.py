"""Corpus metadata heuristics."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.adapters.legal_corpus.discovery import scan_corpus
from app.ingestion.adapters.legal_corpus.metadata import infer_corpus_metadata


def _candidate(root: Path, rel: str):
    found = {c.relative_path: c for c in scan_corpus(root)}
    return found[rel]


def test_infer_jurisdiction_es_and_eu(tmp_path) -> None:
    root = tmp_path / "legalize_sample"
    (root / "legalize-es").mkdir(parents=True)
    (root / "legalize-es" / "doc.md").write_text("# x\n", encoding="utf-8")
    (root / "legalize-eu").mkdir(parents=True)
    (root / "legalize-eu" / "doc.md").write_text("# x\n", encoding="utf-8")

    c_es = _candidate(root, "legalize-es/doc.md")
    m_es = infer_corpus_metadata(root, c_es, {})
    assert m_es["jurisdiction"] == "es"

    c_eu = _candidate(root, "legalize-eu/doc.md")
    m_eu = infer_corpus_metadata(root, c_eu, {})
    assert m_eu["jurisdiction"] == "eu"


def test_infer_jurisdiction_unknown(tmp_path) -> None:
    root = tmp_path / "corp"
    (root / "legalize-test").mkdir(parents=True)
    (root / "legalize-test" / "doc.md").write_text("# x\n", encoding="utf-8")
    c = _candidate(root, "legalize-test/doc.md")
    assert infer_corpus_metadata(root, c, {})["jurisdiction"] == "unknown"


def test_infer_document_type_from_filename(tmp_path) -> None:
    root = tmp_path / "legalize_sample"
    (root / "legalize-es").mkdir(parents=True)
    (root / "legalize-es" / "some-regulation-topic.md").write_text("# x\n", encoding="utf-8")
    c = _candidate(root, "legalize-es/some-regulation-topic.md")
    assert infer_corpus_metadata(root, c, {})["legal_document_type"] == "regulation"


def test_frontmatter_overrides_heuristics(tmp_path) -> None:
    root = tmp_path / "legalize_sample"
    (root / "legalize-es").mkdir(parents=True)
    (root / "legalize-es" / "doc.md").write_text("# x\n", encoding="utf-8")
    c = _candidate(root, "legalize-es/doc.md")
    fm = {"jurisdiction": "eu", "document_type": "directive", "language": "en"}
    meta = infer_corpus_metadata(root, c, fm)
    assert meta["jurisdiction"] == "eu"
    assert meta["legal_document_type"] == "directive"
    assert meta["language"] == "en"
