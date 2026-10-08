"""Structured logging utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


class _ConsoleFormatter(logging.Formatter):
    """Simple formatter with level prefix."""

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.fmt, datefmt=self.datefmt)


def get_logger(
    name: str, log_file: Path | str | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Return a structured logger with optional file handler.

    Args:
        name: Logger name.
        log_file: Optional file path to also log to.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if the same name is requested multiple times.
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(_ConsoleFormatter())
        logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Check if a file handler already points here.
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(
                log_path.resolve()
            ):
                return logger

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(_ConsoleFormatter())
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def log_dict(
    logger: logging.Logger, prefix: str, data: dict[str, Any], level: int = logging.INFO
) -> None:
    """Log a dictionary in a readable key=value format."""
    for key, value in data.items():
        logger.log(level, "%s%s=%s", prefix, key, value)
