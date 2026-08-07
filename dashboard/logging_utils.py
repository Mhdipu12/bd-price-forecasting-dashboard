"""Logging configuration shared by the Streamlit app and the export scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from dashboard.config import LOG_DIR

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s"


def configure_logging(level: int = logging.INFO, log_file: str = "dashboard.log") -> None:
    """Attach a console and a rotating-free file handler exactly once.

    Repeated calls are no-ops, which matters in Streamlit where the whole
    script re-executes on every interaction.

    Args:
        level: Root level for the ``dashboard`` logger hierarchy.
        log_file: File name created inside ``logs/``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("dashboard")
    root.setLevel(level)
    root.propagate = False

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(LOG_DIR) / log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(file_handler)
    except OSError:  # read-only deployment target — console logging still works
        root.warning("Could not open a log file in %s; logging to console only.", LOG_DIR)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring the hierarchy on first use.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A logger under the ``dashboard`` namespace.
    """
    configure_logging()
    short = name.split(".")[-1]
    return logging.getLogger(f"dashboard.{short}")
