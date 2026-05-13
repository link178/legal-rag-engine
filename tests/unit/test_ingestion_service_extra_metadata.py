"""IngestionService extra_metadata merge."""

from __future__ import annotations

from app.ingestion.services import default_ingestion_service


def test_ingest_file_extra_metadata_merged(tmp_path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("hello\n", encoding="utf-8")
    svc = default_ingestion_service()
    doc = svc.ingest_file(
        p,
        extra_metadata={
            "corpus_name": "demo",
            "file_checksum_sha256": "must-not-win",
        },
    )
    assert doc.metadata["corpus_name"] == "demo"
    assert doc.metadata["file_checksum_sha256"] != "must-not-win"
