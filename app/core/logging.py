"""Logging setup."""

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once at application startup."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
