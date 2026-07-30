"""Alembic environment: load ``Settings.database_url`` and ORM metadata."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context

# noqa: F401 - register mapped classes on Base.metadata
from app.storage.postgres import models as _postgres_models  # noqa: F401
from app.storage.postgres.base import Base
from sqlalchemy import engine_from_config, pool

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_database_url() -> str:
    """Database URL from application settings (respects ``DATABASE_URL`` / ``.env``)."""
    from app.core.config import get_settings

    return get_settings().database_url


def _sync_config_url_from_settings() -> None:
    """Override ``sqlalchemy.url`` in alembic.ini with the runtime DSN."""
    config.set_main_option("sqlalchemy.url", get_database_url())


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    _sync_config_url_from_settings()
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    _sync_config_url_from_settings()
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
