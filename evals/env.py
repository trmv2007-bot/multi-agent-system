"""Process defaults an eval run must not inherit from the machine it runs on.

An eval run has to be repeatable: no run state from an earlier case, no trace
export, no desktop control. These are defaults, so an environment that sets a
value keeps it.
"""

import os

DEFAULTS = {
    "CHECKPOINT_DB": "",
    "OTEL_ENABLED": "false",
    "COMPUTER_USE": "false",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_TRACING": "false",
}
"""Name to value, applied only where the environment has nothing."""


def isolate() -> None:
    """Apply the defaults. Call before importing anything from ``src``."""
    for name, value in DEFAULTS.items():
        os.environ.setdefault(name, value)
