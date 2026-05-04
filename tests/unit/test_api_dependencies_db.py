"""Test get_session dependency yields and closes without real DB."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from app.api.dependencies.db import get_session
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient


def test_get_session_yield_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    class _FakeSession:
        pass

    class _Ctx:
        def __enter__(self):
            return _FakeSession()

        def __exit__(self, *args):
            closed.append(True)
            return None

    monkeypatch.setattr(
        "app.api.dependencies.db.session_scope",
        lambda database_url=None: _Ctx(),
    )
    monkeypatch.setattr(
        "app.api.dependencies.db.get_settings",
        lambda: MagicMock(database_url="postgresql://test/test"),
    )

    app = FastAPI()

    @app.get("/x")
    def _x(session=Depends(get_session)):
        assert session.__class__.__name__ == "_FakeSession"
        return {"ok": True}

    c = TestClient(app)
    r = c.get("/x")
    assert r.status_code == 200
    assert closed == [True]
