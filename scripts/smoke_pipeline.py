#!/usr/bin/env python3
"""End-to-end smoke pipeline for local Postgres (operator-controlled).

Prerequisites (not started by this script):

- ``docker compose up -d postgres`` (or equivalent)
- ``alembic upgrade head``
- ``pip install -e ".[dev,demo]"`` from repo root

Steps:

- Ingest ``data/sample_corpus/basic`` files (required for eval goldens).
- Import ``data/sample_corpus/legalize_sample`` (legal adapter demo).
- Chunk all persisted documents (``fixed_size``).
- Index (``fixed_size`` manifest; dense + sparse on by default).
- Hybrid retrieval + mock generation on an EU demo question.
- Retrieval + answer evaluation JSONL (answer eval uses ``sparse_only`` for golden q3).

Exit code is non-zero if any subprocess fails or golden rates are zero.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_json(cmd: list[str], *, cwd: Path) -> dict[str, object]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "(no output)").strip()
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    if not out:
        raise RuntimeError(f"empty stdout from: {' '.join(cmd)}")
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid JSON from {' '.join(cmd)}: {e}\n{out[:4000]}") from e


def _require_no_error(payload: dict[str, object], *, step: str) -> None:
    err = payload.get("error")
    if err:
        raise RuntimeError(f"{step}: {err!r}\n{json.dumps(payload, indent=2)[:4000]}")


def _collect_document_ids(*payloads: dict[str, object]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    for payload in payloads:
        did = payload.get("document_id")
        if isinstance(did, str) and did and did not in seen:
            seen.add(did)
            ids.append(did)

    return ids


def _dedupe_preserve_order(ids: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for d in ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _document_ids_from_corpus_summary(summary: dict[str, object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in summary.get("items", []):
        if not isinstance(item, dict):
            continue
        if item.get("status") == "failed":
            continue
        if item.get("error"):
            continue
        did = item.get("document_id")
        if isinstance(did, str) and did and did not in seen:
            seen.add(did)
            out.append(did)
    return out


def chunk_documents(py: str, doc_ids: Iterable[str], *, cwd: Path) -> None:
    for did in doc_ids:
        payload = run_json(
            [
                py,
                "-m",
                "app.chunking.cli",
                did,
                "--strategy",
                "fixed_size",
                "--json",
            ],
            cwd=cwd,
        )
        _require_no_error(payload, step=f"chunk {did}")


def _require_listening_tcp(host: str, port: int, *, timeout: float = 2.0) -> None:
    """Fail fast when Postgres is not accepting connections (common operator mistake)."""
    with socket.create_connection((host, port), timeout=timeout):
        pass


def main(argv: list[str] | None = None) -> int:
    _ = argv
    try:
        _require_listening_tcp("127.0.0.1", 5432, timeout=2.0)
    except OSError as e:
        print(
            "error: cannot reach 127.0.0.1:5432 (PostgreSQL). "
            "Run `docker compose up -d postgres` and `alembic upgrade head`, then retry.\n"
            f"Details: {e}",
            file=sys.stderr,
        )
        return 1

    root = _repo_root()
    py = sys.executable
    basic_dir = root / "data" / "sample_corpus" / "basic"
    legal_root = root / "data" / "sample_corpus" / "legalize_sample"

    print("== Ingest basic sample (eval goldens) ==")
    intro = basic_dir / "intro.md"
    plain = basic_dir / "plain.txt"
    intro_p = run_json(
        [py, "-m", "app.ingestion.cli", str(intro), "--persist", "--json"],
        cwd=root,
    )
    _require_no_error(intro_p, step="ingest intro.md")
    plain_p = run_json(
        [py, "-m", "app.ingestion.cli", str(plain), "--persist", "--json"],
        cwd=root,
    )
    _require_no_error(plain_p, step="ingest plain.txt")

    print("== Legal corpus import ==")
    legal_summary = run_json(
        [
            py,
            "-m",
            "app.ingestion.adapters.legal_corpus.cli",
            str(legal_root),
            "--persist",
            "--json",
        ],
        cwd=root,
    )
    if int(legal_summary.get("failed_count") or 0) > 0:
        print(json.dumps(legal_summary, indent=2), file=sys.stderr)
        return 1

    doc_ids = _collect_document_ids(intro_p, plain_p)
    doc_ids.extend(_document_ids_from_corpus_summary(legal_summary))
    doc_ids = _dedupe_preserve_order(doc_ids)
    if not doc_ids:
        print("no document ids collected", file=sys.stderr)
        return 1

    print("== Chunking (fixed_size) ==")
    chunk_documents(py, doc_ids, cwd=root)

    print("== Indexing ==")
    idx = run_json(
        [py, "-m", "app.indexing.cli", "--chunking-strategy", "fixed_size", "--json"],
        cwd=root,
    )
    _require_no_error(idx, step="indexing")
    mid = idx.get("manifest_id")
    if not isinstance(mid, str) or not mid:
        print(json.dumps(idx, indent=2), file=sys.stderr)
        return 1

    q_demo = "What does the demo AI Act fragment describe?"
    print("== Retrieval (hybrid, EU filter) ==")
    run_json(
        [
            py,
            "-m",
            "app.retrieval.cli",
            q_demo,
            "--mode",
            "hybrid",
            "--chunking-strategy",
            "fixed_size",
            "--index-manifest-id",
            mid,
            "--filter-jurisdiction",
            "eu",
            "--json",
        ],
        cwd=root,
    )

    print("== Generation (mock, hybrid, EU filter) ==")
    gen = run_json(
        [
            py,
            "-m",
            "app.generation.cli",
            q_demo,
            "--mode",
            "hybrid",
            "--chunking-strategy",
            "fixed_size",
            "--provider",
            "mock",
            "--index-manifest-id",
            mid,
            "--filter-jurisdiction",
            "eu",
            "--json",
        ],
        cwd=root,
    )
    if gen.get("insufficient_context"):
        print(json.dumps(gen, indent=2), file=sys.stderr)
        return 1

    print("== Evaluation: retrieval (hybrid) ==")
    ret_eval = run_json(
        [
            py,
            "-m",
            "app.evaluation.cli",
            "retrieval",
            str(root / "data" / "eval" / "retrieval_golden.jsonl"),
            "--mode",
            "hybrid",
            "--chunking-strategy",
            "fixed_size",
            "--index-manifest-id",
            mid,
            "--json",
        ],
        cwd=root,
    )

    print("== Evaluation: answer (sparse_only; golden q3) ==")
    ans_eval = run_json(
        [
            py,
            "-m",
            "app.evaluation.cli",
            "answer",
            str(root / "data" / "eval" / "answer_golden.jsonl"),
            "--mode",
            "sparse_only",
            "--chunking-strategy",
            "fixed_size",
            "--provider",
            "mock",
            "--index-manifest-id",
            mid,
            "--json",
        ],
        cwd=root,
    )

    rsum = ret_eval.get("summary")
    asum = ans_eval.get("summary")
    hit_rate = rsum.get("hit_rate") if isinstance(rsum, dict) else None
    pass_rate = asum.get("pass_rate") if isinstance(asum, dict) else None

    print("\n== Summary ==")
    print(f"manifest_id={mid}")
    print(f"documents_chunked={len(doc_ids)}")
    print(f"retrieval hit_rate={hit_rate}")
    print(f"answer pass_rate={pass_rate}")

    if hit_rate is None or pass_rate is None:
        return 1
    if float(hit_rate) <= 0 or float(pass_rate) <= 0:
        print("expected non-zero hit_rate and pass_rate", file=sys.stderr)
        print(json.dumps({"retrieval": ret_eval, "answer": ans_eval}, indent=2)[:8000])
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
