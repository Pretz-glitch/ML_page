"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io import load_dataset  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    """Load the raw archive once for the whole test session."""
    return load_dataset(ROOT / "data" / "raw" / "water_dataset.mat")
