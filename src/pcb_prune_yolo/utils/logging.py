"""Logging configuration."""

import logging


def configure_logging(verbose: bool = False) -> None:
    """Configure process-wide console logging."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s: %(message)s")

