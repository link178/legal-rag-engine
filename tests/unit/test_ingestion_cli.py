"""CLI entrypoint (no Postgres)."""

from __future__ import annotations

from app.ingestion.cli import main


def test_cli_ingest_prints_fields(tmp_path, capsys) -> None:
    p = tmp_path / "c.md"
    p.write_text("# H\n\nok\n", encoding="utf-8")
    code = main([str(p)])
    assert code == 0
    out = capsys.readouterr().out
    assert "source_type: markdown" in out
    assert "checksum:" in out


def test_cli_json_mode(tmp_path, capsys) -> None:
    p = tmp_path / "c.txt"
    p.write_text("x", encoding="utf-8")
    code = main([str(p), "--json"])
    assert code == 0
    out = capsys.readouterr().out
    assert '"source_type": "text"' in out
