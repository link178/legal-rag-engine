"""SQLAlchemy declarative base for Postgres mappings."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all Postgres ORM models."""
