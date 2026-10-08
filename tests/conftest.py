"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    """Configure deterministic defaults for tests."""
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="session")
def device() -> str:
    """Return the best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="function", autouse=True)
def seed() -> None:
    """Reset random seed before every test."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
