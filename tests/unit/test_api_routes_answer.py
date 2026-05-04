"""POST /v1/answer route tests (DB-free)."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from app.generation.models import (
    CitationVerificationResult,
    GroundedAnswer,
    GroundedCitation,
)
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("APP_NAME", "legal-rag-engine")
    from app.core.config import get_settings

    get_settings.cache_clear()
    return TestClient(app)


def test_answer_provider_openai_no_db(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    calls: list[str] = []

    def fake_scope(database_url: str | None = None):
        class _CM:
            def __enter__(self):
                calls.append("entered")
                raise AssertionError("session_scope should not run for bad provider")

            def __exit__(self, *_a):
                return None

        return _CM()

    monkeypatch.setattr("app.api.dependencies.answer.session_scope", fake_scope)

    r = client.post(
        "/v1/answer",
        json={"question": "q?", "provider": "openai"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_provider"
    assert "openai" in r.json()["error"]["message"]
    assert calls == []


def test_answer_empty_question_422(client: TestClient) -> None:
    r = client.post("/v1/answer", json={"question": ""})
    assert r.status_code == 422


def test_answer_manifest_not_found_404(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.retrieval.errors import ManifestNotFoundError

    class _Ctx:
        def __enter__(self):
            return MagicMock()

        def __exit__(self, *a):
            return None

    monkeypatch.setattr(
        "app.api.dependencies.answer.session_scope",
        lambda database_url=None: _Ctx(),
    )

    monkeypatch.setattr(
        "app.api.routes.answer.GroundedAnswerer.from_session",
        MagicMock(side_effect=ManifestNotFoundError("no index")),
    )

    r = client.post("/v1/answer", json={"question": "why?", "provider": "mock"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "manifest_not_found"


def _fake_ga_cls_factory(out: GroundedAnswer):
    class _GA:
        @classmethod
        def from_session(cls, _session, _cfg, context_builder=None, provider=None):
            class _I:
                def answer(self, _q: str):
                    return out

            return _I()

    return _GA


def test_answer_grounded_shape(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    cid, did = uuid4(), uuid4()
    cite = GroundedCitation(
        citation_id=1,
        chunk_id=cid,
        document_id=did,
        source_path="p.md",
        title=None,
        heading=None,
        rank=1,
        score=0.5,
        text_preview="pv",
    )
    out = GroundedAnswer(
        question="q",
        answer="a [1]",
        mode="grounded",
        citations=(cite,),
        used_citation_ids=(1,),
        retrieval_mode="hybrid",
        insufficient_context=False,
        metadata={"k": 1},
        citation_verification=CitationVerificationResult(
            used_citation_ids=(1,),
            available_citation_ids=(1,),
            valid_citation_ids=(1,),
            invalid_citation_ids=(),
            unused_citation_ids=(),
            duplicate_citation_ids=(),
            citation_validity_rate=1.0,
            has_citations=True,
            has_valid_citations=True,
            has_invalid_citations=False,
        ),
    )
    monkeypatch.setattr("app.api.routes.answer.GroundedAnswerer", _fake_ga_cls_factory(out))

    class _Ctx:
        def __enter__(self):
            return MagicMock()

        def __exit__(self, *a):
            return None

    monkeypatch.setattr(
        "app.api.dependencies.answer.session_scope",
        lambda database_url=None: _Ctx(),
    )

    r = client.post("/v1/answer", json={"question": "q", "provider": "mock"})
    assert r.status_code == 200
    b = r.json()
    assert b["mode"] == "grounded"
    assert b["retrieval_mode"] == "hybrid"
    assert b["citations"][0]["citation_id"] == 1
    assert b["used_citation_ids"] == [1]
    assert b["citation_verification"]["valid_citation_ids"] == [1]


def test_answer_partial_and_insufficient(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Ctx:
        def __enter__(self):
            return MagicMock()

        def __exit__(self, *a):
            return None

    monkeypatch.setattr(
        "app.api.dependencies.answer.session_scope",
        lambda database_url=None: _Ctx(),
    )

    partial = GroundedAnswer(
        question="q",
        answer="no brackets",
        mode="partial",
        citations=(
            GroundedCitation(
                citation_id=1,
                chunk_id=uuid4(),
                document_id=uuid4(),
                source_path=None,
                title=None,
                heading=None,
                rank=1,
                score=0.1,
                text_preview="x",
            ),
        ),
        used_citation_ids=(),
        retrieval_mode="dense_only",
        insufficient_context=False,
        metadata={},
        citation_verification=CitationVerificationResult(
            used_citation_ids=(),
            available_citation_ids=(1,),
            valid_citation_ids=(),
            invalid_citation_ids=(),
            unused_citation_ids=(1,),
            duplicate_citation_ids=(),
            citation_validity_rate=0.0,
            has_citations=False,
            has_valid_citations=False,
            has_invalid_citations=False,
        ),
    )
    monkeypatch.setattr("app.api.routes.answer.GroundedAnswerer", _fake_ga_cls_factory(partial))
    r1 = client.post("/v1/answer", json={"question": "q", "provider": "mock"})
    assert r1.status_code == 200
    assert r1.json()["mode"] == "partial"

    ins = GroundedAnswer(
        question="q",
        answer="I do not have enough information in the provided context to answer this question.",
        mode="insufficient_context",
        citations=(),
        used_citation_ids=(),
        retrieval_mode="hybrid",
        insufficient_context=True,
        metadata={},
        citation_verification=CitationVerificationResult(
            used_citation_ids=(),
            available_citation_ids=(),
            valid_citation_ids=(),
            invalid_citation_ids=(),
            unused_citation_ids=(),
            duplicate_citation_ids=(),
            citation_validity_rate=0.0,
            has_citations=False,
            has_valid_citations=False,
            has_invalid_citations=False,
        ),
    )
    monkeypatch.setattr("app.api.routes.answer.GroundedAnswerer", _fake_ga_cls_factory(ins))
    r2 = client.post("/v1/answer", json={"question": "q2", "provider": "mock"})
    assert r2.status_code == 200
    assert r2.json()["mode"] == "insufficient_context"
    assert r2.json()["insufficient_context"] is True
