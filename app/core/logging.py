"""Configure stdlib logging from application settings."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import Settings


_CONFIGURED = False


def configure_logging(settings: Settings) -> None:
    """Set log level and a simple readable format; safe for local dev and pytest."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = getattr(settings, "log_level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)

    # Reduce noise from third-party libs in development
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _CONFIGURED = True


def reset_logging_for_tests() -> None:
    """Allow tests to re-run configure_logging with different settings."""
    global _CONFIGURED
    _CONFIGURED = False
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
