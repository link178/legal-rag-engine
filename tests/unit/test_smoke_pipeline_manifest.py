"""Smoke pipeline passes the created manifest into evaluation commands."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from uuid import uuid4

import pytest


def _load_smoke_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "smoke_pipeline.py"
    spec = importlib.util.spec_from_file_location("smoke_pipeline_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def smoke() -> ModuleType:
    return _load_smoke_module()


def _fake_run_json_factory(manifest_id: str):
    calls: list[list[str]] = []

    def _run_json(cmd: list[str], *, cwd: Path) -> dict[str, object]:
        calls.append(list(cmd))
        joined = " ".join(cmd)
        if "app.ingestion.cli" in joined and "legal_corpus" not in joined:
            return {"document_id": str(uuid4()), "error": None}
        if "legal_corpus" in joined:
            return {
                "failed_count": 0,
                "items": [{"document_id": str(uuid4()), "status": "ok"}],
            }
        if "app.chunking.cli" in joined:
            return {"error": None}
        if "app.indexing.cli" in joined:
            return {"manifest_id": manifest_id, "error": None}
        if "app.retrieval.cli" in joined:
            return {"results": []}
        if "app.generation.cli" in joined:
            return {"insufficient_context": False, "answer": "ok"}
        if "app.evaluation.cli" in joined and "retrieval" in joined:
            return {
                "manifest": {"id": manifest_id},
                "summary": {
                    "hit_rate": 1.0,
                    "execution_status": "PASSED",
                    "total_questions": 9,
                },
            }
        if "app.evaluation.cli" in joined and "answer" in joined:
            return {
                "manifest": {"id": manifest_id},
                "summary": {
                    "pass_rate": 1.0,
                    "execution_status": "PASSED",
                    "total_questions": 10,
                },
            }
        raise AssertionError(f"unexpected command: {joined}")

    return _run_json, calls


def test_smoke_passes_manifest_to_retrieval_and_answer_eval(
    smoke: ModuleType, tmp_path: Path
) -> None:
    mid = str(uuid4())
    fake_run, calls = _fake_run_json_factory(mid)
    manifest_out = tmp_path / "manifest_id.txt"

    with patch.object(smoke, "_require_listening_tcp"):
        with patch.object(smoke, "run_json", side_effect=fake_run):
            rc = smoke.main(["--manifest-out", str(manifest_out)])

    assert rc == 0
    assert manifest_out.read_text(encoding="utf-8").strip() == mid

    ret_cmds = [c for c in calls if "app.evaluation.cli" in c and "retrieval" in c]
    ans_cmds = [c for c in calls if "app.evaluation.cli" in c and "answer" in c]
    assert len(ret_cmds) == 1
    assert len(ans_cmds) == 1

    ret = ret_cmds[0]
    ans = ans_cmds[0]
    assert "--index-manifest-id" in ret
    assert ret[ret.index("--index-manifest-id") + 1] == mid
    assert "--index-manifest-id" in ans
    assert ans[ans.index("--index-manifest-id") + 1] == mid


def test_smoke_empty_manifest_id_fails(smoke: ModuleType) -> None:
    calls: list[list[str]] = []

    def _run_json(cmd: list[str], *, cwd: Path) -> dict[str, object]:
        calls.append(list(cmd))
        joined = " ".join(cmd)
        if "app.ingestion.cli" in joined and "legal_corpus" not in joined:
            return {"document_id": str(uuid4()), "error": None}
        if "legal_corpus" in joined:
            return {
                "failed_count": 0,
                "items": [{"document_id": str(uuid4()), "status": "ok"}],
            }
        if "app.chunking.cli" in joined:
            return {"error": None}
        if "app.indexing.cli" in joined:
            return {"manifest_id": "", "error": None}
        raise AssertionError(f"unexpected command: {joined}")

    with patch.object(smoke, "_require_listening_tcp"):
        with patch.object(smoke, "run_json", side_effect=_run_json):
            rc = smoke.main([])

    assert rc == 1
    assert not any("app.evaluation.cli" in c for c in calls)


def test_smoke_manifest_out_optional(smoke: ModuleType, tmp_path: Path) -> None:
    mid = str(uuid4())
    fake_run, _calls = _fake_run_json_factory(mid)

    with patch.object(smoke, "_require_listening_tcp"):
        with patch.object(smoke, "run_json", side_effect=fake_run):
            rc = smoke.main([])

    assert rc == 0
    assert not (tmp_path / "manifest_id.txt").exists()
