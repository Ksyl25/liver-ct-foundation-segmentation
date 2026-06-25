"""Simple logger setup with an optional Rich handler."""

from __future__ import annotations

import logging


def get_logger(name: str = "liver_ct_foundation_segmentation") -> logging.Logger:
    """Return a configured project logger."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(rich_tracebacks=True)
    except ImportError:
        handler = logging.StreamHandler()

    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger
