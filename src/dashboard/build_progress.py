"""The progress one page build reports while it runs.

A build is long: a planner, research, one writer per section, a review, and the
sections it sends back for repair. The dashboard shows the work as a status line
and a checklist of the planned sections, so a person can tell what is happening
and when it will stop without watching a spinner.

The states below are fed only by events the run already produces: the workflow's
``state`` snapshots (the plan, the review's validation results) and the activity
bus (``worker.N`` reporting, and each finished ``section``). Nothing here invents
a second event stream, and nothing here calls a model.
"""

from dataclasses import dataclass, field
from typing import Any

PENDING = "pending"
WRITING = "writing"
DONE = "done"


@dataclass
class BuildProgress:
    """What one build has planned, written, reviewed and finished."""

    plan: list[str] = field(default_factory=list)
    items: dict[int, str] = field(default_factory=dict)
    phase: str = "planning"
    status: str = "planning"
    sections: int = 0
    started: float = 0.0
    seconds: float = 0.0
    reviewed: bool = False

    # ── what the run reports ────────────────────────────────────────
    def saw_plan(self, names: list[str]) -> None:
        """The planner decided the page's sections; show them as pending."""
        if not names or list(names) == self.plan:
            return
        self.plan = [str(name) for name in names]
        for index in range(len(self.plan)):
            self.items.setdefault(index, PENDING)
        if self.phase == "planning":
            self.phase = "writing"
            self.status = f"planned {_plural(len(self.plan), 'section')}"

    def saw_worker(self, index: int) -> None:
        """One section writer started on its section."""
        self.items[index] = WRITING
        self.phase = "fixing" if self.reviewed else "writing"
        verb = "fixing" if self.reviewed else "writing"
        total = len(self.plan) or index + 1
        self.status = f"{verb} section {index + 1} of {total}"

    def saw_section(self, index: int) -> None:
        """A section landed: a first draft, or a repair the review asked for."""
        self.items[index] = DONE
        self.sections = self.count
        total = len(self.plan)
        if self.reviewed:
            self.phase = "fixing"
            self.status = f"fixed section {index + 1} of {total}" if total else "fixing"
            return
        if self.sections >= total:
            self.phase = "reviewing"
            self.status = "reviewing"
            return
        self.phase = "writing"
        ahead = self.next_pending(index)
        self.status = f"writing section {ahead + 1} of {total}" if ahead is not None else "writing"

    def saw_review(self) -> None:
        """The review returned; anything written from here is a repair."""
        if self.phase == "built":
            return
        self.reviewed = True
        self.phase = "reviewing"
        self.status = "reviewing"

    def saw_artifact(self, sections: int) -> None:
        """The preview grew: this many sections are on the page."""
        self.sections = max(self.sections, sections)

    # ── how it ended ───────────────────────────────────────────────
    def built(self, seconds: float) -> None:
        """The page is finished; say how much of it and how long it took."""
        self.seconds = seconds
        self.phase = "built"
        self.sections = max(self.sections, self.count)
        self.status = f"built {_plural(self.sections, 'section')} in {round(seconds)}s"

    def stopped(self, seconds: float) -> None:
        """The build stopped early: say how far it got."""
        self.seconds = seconds
        self.phase = "failed"
        self.status = (
            f"stopped after {self.sections} of {len(self.plan)} sections"
            if self.plan
            else "stopped before any section was written"
        )

    # ── what the page draws ────────────────────────────────────────
    @property
    def count(self) -> int:
        """How many sections are done."""
        return sum(1 for state in self.items.values() if state == DONE)

    @property
    def running(self) -> bool:
        """Whether the build is still going."""
        return self.phase not in {"built", "failed"}

    def checklist(self) -> list[dict[str, Any]]:
        """Every planned section and what has happened to it."""
        return [
            {"index": index, "name": name, "state": self.items.get(index, PENDING)}
            for index, name in enumerate(self.plan)
        ]

    def next_pending(self, after: int) -> int | None:
        """The next section still to be written, preferring the ones after ``after``."""
        for index in range(after + 1, len(self.plan)):
            if self.items.get(index) != DONE:
                return index
        for index in range(len(self.plan)):
            if self.items.get(index) != DONE:
                return index
        return None

    def as_dict(self) -> dict[str, Any]:
        """The progress, ready to be sent to the page."""
        return {
            "plan": list(self.plan),
            "checklist": self.checklist(),
            "phase": self.phase,
            "status": self.status,
            "sections": self.sections,
            "seconds": self.seconds,
            "reviewed": self.reviewed,
            "running": self.running,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuildProgress":
        """Rebuild progress saved to disk, so a refresh keeps the checklist."""
        items = {int(index): str(state) for index, state in (data.get("items") or {}).items()}
        return cls(
            plan=[str(name) for name in data.get("plan") or []],
            items=items,
            phase=str(data.get("phase") or "planning"),
            status=str(data.get("status") or "planning"),
            sections=int(data.get("sections") or 0),
            started=float(data.get("started") or 0.0),
            seconds=float(data.get("seconds") or 0.0),
            reviewed=bool(data.get("reviewed")),
        )


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"
