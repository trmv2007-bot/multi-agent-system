"""The scripted model the dashboard and the browser test run on.

The provider itself is :mod:`src.demo` — the same script `multi-agent-demo` serves
— so the page, the browser test and the demo cannot drift apart. Each agent's
system prompt is distinct, and the script answers each one with the shape its
parser expects.

``DASHBOARD_FAKE_MODEL`` turns it on for one test run without setting the
app-wide ``DEMO_MODE``; nothing in a normal run sets either.
"""

import os

from src.config import settings
from src.demo import DEMO_ANSWER, DemoModel, demo_model

ENV_FLAG = "DASHBOARD_FAKE_MODEL"
"""Set to 1 to answer the dashboard's turns from the script."""

_TRUTHY = {"1", "true", "yes", "on"}

ANSWER = DEMO_ANSWER
"""What the writer, synthesizer and reporter paths return, so a test can assert on it."""


def scripted_enabled() -> bool:
    """Whether this process should answer from the script."""
    return os.environ.get(ENV_FLAG, "").strip().lower() in _TRUTHY or settings.demo_mode


def scripted_llm() -> DemoModel:
    """The scripted provider."""
    return demo_model()
