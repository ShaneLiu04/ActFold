"""Tests for actfold.utils.logger."""

from __future__ import annotations

import logging
from pathlib import Path

from actfold.utils.logger import get_logger, log_dict


def test_get_logger_returns_logger() -> None:
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO


def test_get_logger_with_file(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    logger = get_logger("test_logger_file", log_file=log_file)
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    logger.info("test message")
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "test message" in content


def test_log_dict(tmp_path: Path) -> None:
    log_file = tmp_path / "dict.log"
    logger = get_logger("test_log_dict_file", log_file=log_file)
    log_dict(logger, "cfg.", {"a": 1, "b": 2})
    content = log_file.read_text(encoding="utf-8")
    assert "cfg.a=1" in content
    assert "cfg.b=2" in content
