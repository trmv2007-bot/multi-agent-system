"""Shared test setup.

Tests must never write to the project's checkpoint database, or one run's
thread would leak into the next one's assertions.
"""

import pytest

from src.activity import activity
from src.config import settings


@pytest.fixture(autouse=True)
def isolated_checkpoints(monkeypatch):
    """Default every test to the in-memory checkpointer and a clean activity bus."""
    monkeypatch.setattr(settings, "checkpoint_db", None)
    activity.reset()
    yield
    activity.reset()
