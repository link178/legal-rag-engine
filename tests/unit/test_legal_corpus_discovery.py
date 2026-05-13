"""Legal corpus discovery rules."""

from __future__ import annotations

import sys

import pytest
from app.ingestion.adapters.legal_corpus.discovery import scan_corpus


def test_scan_finds_supported_extensions(tmp_path) -> None:
    root = tmp_path / "corp"
    (root / "legalize-es").mkdir(parents=True)
    (root / "legalize-es" / "a.md").write_text("# x\n", encoding="utf-8")
    (root / "legalize-es" / "b.markdown").write_text("# x\n", encoding="utf-8")
    (root / "legalize-es" / "c.txt").write_text("x", encoding="utf-8")
    (root / "legalize-es" / "d.html").write_text(
        "<html><body><p>h</p></body></html>",
        encoding="utf-8",
    )
    (root / "legalize-es" / "e.htm").write_text(
        "<html><body><p>h</p></body></html>",
        encoding="utf-8",
    )
    (root / "legalize-es" / "f.pdf").write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    found = scan_corpus(root)
    rels = [c.relative_path for c in found]
    assert rels == sorted(rels)
    assert set(rels) == {
        "legalize-es/a.md",
        "legalize-es/b.markdown",
        "legalize-es/c.txt",
        "legalize-es/d.html",
        "legalize-es/e.htm",
        "legalize-es/f.pdf",
    }


def test_scan_ignores_dirs_and_extensions(tmp_path) -> None:
    root = tmp_path / "corp"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".git" / "x.md").write_text("nope", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "y.md").write_text("nope", encoding="utf-8")
    (root / "keep").mkdir()
    (root / "keep" / "ok.md").write_text("# OK\n", encoding="utf-8")
    (root / "keep" / "bad.docx").write_bytes(b"x")

    found = scan_corpus(root)
    assert [c.relative_path for c in found] == ["keep/ok.md"]


def test_scan_skips_hidden_files(tmp_path) -> None:
    root = tmp_path / "corp"
    root.mkdir()
    (root / "a.md").write_text("# x\n", encoding="utf-8")
    (root / ".secret.md").write_text("# x\n", encoding="utf-8")

    found = scan_corpus(root)
    assert [c.relative_path for c in found] == ["a.md"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlink privileges vary on Windows")
def test_scan_skips_symlinks(tmp_path) -> None:
    root = tmp_path / "corp"
    root.mkdir()
    (root / "t.txt").write_text("hello", encoding="utf-8")
    link = root / "linked.txt"
    try:
        link.symlink_to(root / "t.txt")
    except OSError:
        pytest.skip("cannot create symlink")

    found = scan_corpus(root)
    assert [c.relative_path for c in found] == ["t.txt"]


def test_scan_extension_case_insensitive(tmp_path) -> None:
    root = tmp_path / "corp"
    root.mkdir()
    (root / "a.MD").write_text("# x\n", encoding="utf-8")
    found = scan_corpus(root)
    assert len(found) == 1
    assert found[0].relative_path == "a.MD"


def test_scan_empty_dir(tmp_path) -> None:
    root = tmp_path / "corp"
    root.mkdir()
    assert scan_corpus(root) == ()
