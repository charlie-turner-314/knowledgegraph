from __future__ import annotations

import logging
from typing import Optional

from .config import settings


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging only once."""

    logging.basicConfig(
        level=(level or settings.log_level).upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


configure_logging()
