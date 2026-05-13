"""Optional DB integration tests (skipped unless opted in).

Set ``LEGAL_RAG_RUN_INTEGRATION_DB=1`` and a reachable ``DATABASE_URL`` before
adding real connectivity or Alembic checks here.
"""

from __future__ import annotations

import os

import pytest

RUN_DB_INTEGRATION = os.environ.get("LEGAL_RAG_RUN_INTEGRATION_DB") == "1"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not RUN_DB_INTEGRATION,
    reason="Set LEGAL_RAG_RUN_INTEGRATION_DB=1 to opt into integration DB tests",
)
def test_placeholder_opt_in() -> None:
    """Replace with engine.connect() or Alembic checks when CI has Postgres."""
    assert True
