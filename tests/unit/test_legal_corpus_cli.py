"""CLI for legal corpus import."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.adapters.legal_corpus.cli import main


def test_cli_json_mode(tmp_path, capsys) -> None:
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    code = main([str(tmp_path), "--json"])
    assert code == 0
    out = capsys.readouterr().out
    assert '"corpus_name"' in out
    assert '"discovered_count"' in out


def test_cli_missing_directory(tmp_path, capsys) -> None:
    missing = tmp_path / "missing_corpus"
    code = main([str(missing)])
    assert code == 1
    err = capsys.readouterr().err
    assert "error:" in err


def test_cli_not_a_directory(tmp_path, capsys) -> None:
    p = tmp_path / "f.txt"
    p.write_text("x", encoding="utf-8")
    code = main([str(p)])
    assert code == 1


def test_cli_markdown_report(tmp_path) -> None:
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    report = tmp_path / "rep.md"
    code = main([str(tmp_path), "--markdown-report", str(report)])
    assert code == 0
    text = report.read_text(encoding="utf-8")
    assert "# Legal Corpus Import Report" in text
    assert "| Status |" in text


def test_cli_persist_empty_corpus_exits(tmp_path, capsys) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    code = main([str(empty), "--persist", "--json"])
    assert code == 1


def test_cli_fail_fast_nonzero_exit(tmp_path) -> None:
    root = tmp_path / "corp"
    root.mkdir()
    (root / "good.txt").write_text("ok\n", encoding="utf-8")
    (root / "bad.txt").write_bytes(b"\xff\xfe")
    code = main([str(root), "--fail-fast"])
    assert code == 1


def test_cli_sample_corpus_smoke_json() -> None:
    repo = Path(__file__).resolve().parents[2]
    sample = repo / "data" / "sample_corpus" / "legalize_sample"
    code = main([str(sample), "--json"])
    assert code == 0
