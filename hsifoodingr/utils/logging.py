from __future__ import annotations

import logging
from typing import Optional

try:
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except Exception:  # pragma: no cover
    RichHandler = None  # type: ignore
    RICH_AVAILABLE = False


def setup_rich_logging(level: int = logging.INFO) -> None:
    """Configure root logger with Rich if available, else basicConfig."""
    if RICH_AVAILABLE and RichHandler is not None:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="%H:%M:%S",
            handlers=[RichHandler(rich_tracebacks=True, show_time=True, markup=True)],
        )
    else:
        logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a configured logger. Call setup_rich_logging() once at startup."""
    return logging.getLogger(name if name else __name__)
