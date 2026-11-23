"""The activity bus: what every agent is doing, right now.

Agents publish a status line whenever they start a model call, run a tool, or
finish. Two consumers listen: the on-screen cursor overlay and the web
dashboard. Publishing costs nothing when nobody is listening.
"""

import atexit
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ActivityEvent = dict[str, Any]


@dataclass
class ToolReport:
    """One tool call, as the dashboard needs to see it."""

    key: str
    label: str
    name: str
    arguments: dict[str, Any]
    ok: bool
    preview: str = ""
    seconds: float = 0.0


logger = logging.getLogger(__name__)


class AgentActivity:
    """Thread-safe channel agents publish activity to."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[ActivityEvent], None]] = []
        self._lock = threading.Lock()
        self._agents: dict[str, dict[str, Any]] = {}
        self._sequence = 0
        self._overlay: Any = None

    # ── publishing ───────────────────────────────────────────────────
    def update(self, key: str, label: str, status: str) -> None:
        """Report what one agent is doing, and notify every listener."""
        with self._lock:
            self._sequence += 1
            entry = self._agents.setdefault(
                key, {"key": key, "label": label, "status": "", "seq": 0}
            )
            entry["label"] = label or entry["label"]
            entry["status"] = status
            entry["seq"] = self._sequence
            event: ActivityEvent = {
                "type": "agent",
                "key": key,
                "label": entry["label"],
                "status": status,
                "seq": self._sequence,
            }
        self._publish(event)

    def agent_started(self, key: str, label: str, role: str = "") -> None:
        """Announce a new agent joining the run."""
        with self._lock:
            self._sequence += 1
            self._agents[key] = {
                "key": key,
                "label": label,
                "status": "waiting",
                "seq": self._sequence,
            }
            event: ActivityEvent = {
                "type": "agent_added",
                "key": key,
                "label": label,
                "role": role,
                "seq": self._sequence,
            }
        self._publish(event)

    def tool(self, report: "ToolReport") -> None:
        """Report a tool call and what it returned, so work is visible as it happens."""
        with self._lock:
            self._sequence += 1
            event: ActivityEvent = {
                "type": "tool",
                "key": report.key,
                "label": report.label,
                "name": report.name,
                "arguments": _short_arguments(report.arguments),
                "ok": report.ok,
                "preview": report.preview[:400],
                "seconds": round(report.seconds, 2),
                "seq": self._sequence,
            }
        self._publish(event)

    def pointer(self, x: float, y: float, label: str = "Agent", status: str = "") -> None:
        """Report where the real mouse pointer is, for the on-screen overlay."""
        with self._lock:
            self._sequence += 1
            event: ActivityEvent = {
                "type": "pointer",
                "x": float(x),
                "y": float(y),
                "label": label,
                "status": status,
                "seq": self._sequence,
            }
        self._publish(event)

    def section(self, index: int, name: str, html: str) -> None:
        """Publish one finished section so a page can be previewed mid-build."""
        with self._lock:
            self._sequence += 1
            event: ActivityEvent = {
                "type": "section",
                "index": index,
                "section": name,
                "html": html,
                "seq": self._sequence,
            }
        self._publish(event)

    def page(self, html: str, title: str = "") -> None:
        """Publish a finished page so the build box can show it."""
        with self._lock:
            self._sequence += 1
            event: ActivityEvent = {
                "type": "page",
                "html": html,
                "title": title,
                "seq": self._sequence,
            }
        self._publish(event)

    def snapshot(self) -> list[dict[str, Any]]:
        """Every agent seen so far, in the order they first appeared."""
        with self._lock:
            return [dict(a) for a in self._agents.values()]

    def reset(self) -> None:
        """Start a fresh run: forget the agents from the previous one."""
        with self._lock:
            self._agents.clear()
            self._sequence = 0

    def _publish(self, event: ActivityEvent) -> None:
        """Hand an event to every listener, isolating failures to the listener."""
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber(event)
            except Exception as e:
                # A broken listener must never break the agent that reported.
                logger.debug("activity listener failed: %s", e)

    # ── subscribing ──────────────────────────────────────────────────
    def subscribe(self, listener: Callable[[ActivityEvent], None]) -> Callable[[], None]:
        """Register a listener and return the function that removes it."""
        with self._lock:
            self._subscribers.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._subscribers:
                    self._subscribers.remove(listener)

        return unsubscribe

    @property
    def listening(self) -> bool:
        """Whether anything is listening for events."""
        with self._lock:
            return bool(self._subscribers)

    # ── the on-screen overlay ────────────────────────────────────────
    def enable(self) -> Any:
        """Start the cursor overlay and subscribe it to this bus."""
        from src.overlay import Overlay

        overlay = Overlay()
        overlay.start()
        self.subscribe(overlay.on_event)
        self._overlay = overlay
        atexit.register(self.disable)
        return overlay

    def disable(self) -> None:
        """Stop the cursor overlay."""
        if self._overlay is not None:
            self._overlay.stop()
            self._overlay = None

    @property
    def enabled(self) -> bool:
        """Whether the cursor overlay is running."""
        return self._overlay is not None


def _short_arguments(arguments: dict[str, Any]) -> str:
    """One readable line describing a tool call's inputs."""
    for key in ("query", "url", "task"):
        if arguments.get(key):
            return str(arguments[key])[:120]
    return ", ".join(f"{k}={v}" for k, v in list(arguments.items())[:2])[:120]


activity = AgentActivity()
