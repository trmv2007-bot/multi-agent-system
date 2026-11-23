"""Live dashboard: a chat interface over the agent team.

Start it with ``multi-agent-dashboard``. Each conversation is a thread: the first
message starts a run, later messages continue it, and the workflow reads the
earlier turns from its checkpointer. Activity (tool calls, agent steps, the build
preview) is attached to each reply but collapsed by default.

The page also carries a comment layer: switch it on, click any part of the UI,
and the note is saved to ``dashboard-notes.json`` so changes can be requested
precisely.
"""

import asyncio
import html
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.activity import activity
from src.artifacts import assemble_page, page_fragments
from src.config import PROJECT_ROOT, settings
from src.dashboard.build_progress import BuildProgress
from src.observability import end_run, start_run

STATIC_DIR = Path(__file__).resolve().parent / "static"
CHAT_FILE = PROJECT_ROOT / "chats.json"
STATE_FILE = PROJECT_ROOT / "thread-state.json"
NOTES_FILE = PROJECT_ROOT / "dashboard-notes.json"
BUILD_FILE = PROJECT_ROOT / "dashboard-builds.json"

POLL_SECONDS = 0.15
KEEPALIVE_SECONDS = 10.0
MAX_RUNS = 40
MAX_THREADS = 40
MAX_BUILDS = 12
TITLE_CHARS = 42
WORKER_KEY = re.compile(r"worker\.(\d+)")


@dataclass
class Run:
    """One turn: a request, the events it produced, and the reply it ended with."""

    id: str
    thread_id: str
    query: str
    mode: str
    subtasks: int
    started: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    answer: str = ""
    error: str = ""
    plan: list[str] = field(default_factory=list)
    text: str = ""
    intensity: str = "normal"
    findings: list[dict[str, Any]] = field(default_factory=list)
    artifact: str = ""
    cost: dict[str, Any] = field(default_factory=dict)
    build_id: str = ""
    fresh: bool = False
    task: asyncio.Task[None] | None = None

    def push(self, event: dict[str, Any]) -> None:
        """Record an event for the browser watching this turn."""
        event["t"] = round(time.time() - self.started, 2)
        self.events.append(event)


@dataclass
class Build:
    """One page: its own tab, its own conversation, its own progress."""

    id: str
    title: str
    thread_id: str
    text: str
    request: str
    document: str = ""
    done: bool = False
    failed: str = ""
    cost: dict[str, Any] = field(default_factory=dict)
    progress: BuildProgress = field(default_factory=BuildProgress)

    def as_dict(self, with_document: bool = True) -> dict[str, Any]:
        """Everything a tab needs to draw itself."""
        payload: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "thread_id": self.thread_id,
            "text": self.text,
            "request": self.request,
            "done": self.done,
            "failed": self.failed,
            "cost": self.cost,
        }
        payload.update(self.progress.as_dict())
        if with_document:
            payload["document"] = self.document
        return payload


RUNS: dict[str, Run] = {}
CHATS: dict[str, list[dict[str, Any]]] = {}
# What each conversation produced: the last page, the research, the summary.
THREAD_STATE: dict[str, dict[str, Any]] = {}
# Each build keeps its page and its progress in its own tab.
BUILDS: dict[str, Build] = {}
BUILD_ORDER: list[str] = []


# ── chat log (display only; the workflow keeps its own memory) ───────
def _load_chats() -> None:
    """Restore the visible conversation log from disk."""
    if not CHAT_FILE.exists():
        return
    try:
        data = json.loads(CHAT_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if isinstance(data, dict):
        for thread_id, messages in data.items():
            if isinstance(messages, list):
                CHATS[thread_id] = messages[:200]


def _save_chats() -> None:
    """Persist the conversation log so a restart keeps the history."""
    try:
        CHAT_FILE.write_text(json.dumps(CHATS, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def _load_builds() -> None:
    """Restore every build's tab: its page, its sections and its progress."""
    if not BUILD_FILE.exists():
        return
    try:
        data = json.loads(BUILD_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(data, list):
        return
    for value in data:
        if not isinstance(value, dict):
            continue
        build = Build(
            id=str(value.get("id") or ""),
            title=str(value.get("title") or "Untitled page"),
            thread_id=str(value.get("thread_id") or ""),
            text=str(value.get("text") or ""),
            request=str(value.get("request") or ""),
            document=str(value.get("document") or ""),
            done=bool(value.get("done")),
            failed=str(value.get("failed") or ""),
            cost=dict(value.get("cost") or {}),
            progress=BuildProgress.from_dict(value),
        )
        if not build.id or not build.thread_id:
            continue
        if not build.done:
            # Nothing can still be running: the server that ran it is gone.
            build.progress.stopped(build.progress.seconds)
            build.done = True
        BUILDS[build.id] = build
        BUILD_ORDER.append(build.id)


def _save_builds() -> None:
    """Persist the tabs so a refresh keeps every page and its checklist."""
    rows = [BUILDS[build_id].as_dict() for build_id in BUILD_ORDER if build_id in BUILDS]
    try:
        BUILD_FILE.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _title_of(text: str) -> str:
    """A tab label: the first few words of what was asked for."""
    words = " ".join(text.split())
    if len(words) <= TITLE_CHARS:
        return words or "Untitled page"
    return words[:TITLE_CHARS].rsplit(" ", 1)[0] + "…"


def _open_build(edit_of: str, text: str) -> tuple[Build, bool]:
    """The tab this build lands in, and whether it starts from nothing.

    A new build gets its own conversation id, so the planner sees no earlier
    page's sections and cannot edit a different page instead of building this
    one. Only an explicit edit of an existing tab carries the sections over.
    """
    existing = BUILDS.get(edit_of.strip()) if edit_of.strip() else None
    if existing is not None:
        sections = existing.progress.plan
        existing.text = text
        existing.request = text
        if sections:
            existing.request = (
                text
                + chr(10)
                + chr(10)
                + "The page currently has these sections: "
                + ", ".join(sections)
                + "."
            )
        # Keep the page on screen while the new version is written.
        existing.done = False
        existing.failed = ""
        existing.cost = {}
        existing.progress = BuildProgress(started=time.time())
        _save_builds()
        return existing, False

    build = Build(
        id=uuid.uuid4().hex[:10],
        title=_title_of(text),
        thread_id="build-" + uuid.uuid4().hex[:10],
        text=text,
        request=text,
        progress=BuildProgress(started=time.time()),
    )
    BUILDS[build.id] = build
    BUILD_ORDER.append(build.id)
    while len(BUILD_ORDER) > MAX_BUILDS:
        BUILDS.pop(BUILD_ORDER.pop(0), None)
    _save_builds()
    return build, True


def _load_state() -> None:
    """Restore each conversation's last page and research."""
    if not STATE_FILE.exists():
        return
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if isinstance(data, dict):
        for thread_id, value in data.items():
            if isinstance(value, dict):
                THREAD_STATE[thread_id] = value


def _save_state() -> None:
    """Persist per-conversation artifacts so a refresh keeps them."""
    try:
        STATE_FILE.write_text(json.dumps(THREAD_STATE, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _remember(  # noqa: PLR0913, PLR0917 - one call site, named arguments only
    thread_id: str,
    mode: str,
    answer: str,
    plan: list[str],
    findings: list[dict],
    artifact: str = "",
) -> None:
    """Record what a turn produced against its conversation."""
    entry = THREAD_STATE.setdefault(thread_id, {})

    if answer.startswith("<!doctype") or answer.lstrip().lower().startswith("<html"):
        # A whole page is an artifact, whichever mode produced it.
        entry["artifact"] = artifact or answer
        entry["build_plan"] = plan
    if findings:
        entry["findings"] = findings[-12:]
    if answer and mode != "build":
        entry["answer"] = answer[:2000]
    entry["mode"] = mode
    _save_state()


def _append(thread_id: str, message: dict[str, Any]) -> None:
    """Add one message to a thread and trim old threads."""
    thread = CHATS.setdefault(thread_id, [])
    thread.append(message)
    if len(CHATS) > MAX_THREADS:
        oldest = sorted(CHATS, key=lambda key: CHATS[key][-1].get("ts", 0))[0]
        CHATS.pop(oldest, None)
    _save_chats()


class ChatRequest(BaseModel):
    """A message from the composer."""

    thread_id: str
    text: str
    mode: str = "team"
    subtasks: int = 3
    intensity: str = "normal"  # normal | deep (research) | super (build)
    build_id: str = ""  # set to edit that page instead of starting a new one


class NoteRequest(BaseModel):
    """A comment pinned to one element of the page."""

    selector: str
    label: str = ""
    text: str
    view: str = "chat"


# ── driving one turn ─────────────────────────────────────────────────
class _BuildCollector:
    """Keeps the build's page and its progress in step with the run."""

    def __init__(self, run: Run) -> None:
        self.run = run
        self.parts: dict[int, str] = {}
        self.theme: dict[str, Any] | None = None
        self.plan_shown = False
        self.build = BUILDS.get(run.build_id) if run.build_id else None

    def _publish(self) -> None:
        """Send the tab its status line and its checklist."""
        if self.build is not None:
            self.run.push({"type": "build", "build": self.build.as_dict(with_document=False)})

    def announce_plan(self) -> None:
        """Show the planned sections as placeholders before any are written."""
        if self.run.mode != "build" or self.plan_shown or self.parts or not self.run.plan:
            return
        self.plan_shown = True
        blocks = [
            (
                '<section class="container" style="opacity:.45">'
                f'<p class="eyebrow">Section {index + 1} of {len(self.run.plan)}</p>'
                f"<h2>{html.escape(str(name))}</h2><p>Writing…</p></section>"
            )
            for index, name in enumerate(self.run.plan)
        ]
        page = assemble_page(self.run.query, blocks, overlay=True)
        if self.build is not None:
            self.build.document = page
        self.run.push({"type": "artifact", "html": page, "sections": 0})
        self._publish()

    def on_state(self, event: dict[str, Any]) -> None:
        """Advance the build from one workflow step, then forward the step."""
        if self.build is None:
            self.run.push(event)
            return

        progress = self.build.progress
        if event.get("plan"):
            progress.saw_plan([str(name) for name in event["plan"]])
            self.run.plan = [str(name) for name in event["plan"]]
        if event.get("validations"):
            progress.saw_review()
        if event.get("sections"):
            progress.saw_artifact(int(event["sections"]))
        if event.get("artifact"):
            self.build.document = str(event["artifact"])

        self.run.push(event)
        self._publish()

    def on_event(self, event: dict[str, Any]) -> None:
        """Forward bus events, turning finished pages into previews."""
        kind = event.get("type")

        if kind == "page":
            self._show_page(str(event.get("html", "")))
            return

        if kind == "section":
            index = int(event["index"])
            self.parts[index] = str(event["html"])
            ordered = [self.parts[key] for key in sorted(self.parts)]
            self._show_page(assemble_page(self.run.query, ordered, overlay=True))
            if self.build is not None:
                self.build.progress.saw_section(index)
            self._publish()
            return

        if kind == "agent" and self.build is not None:
            match = WORKER_KEY.fullmatch(str(event.get("key", "")))
            if match:
                self.build.progress.saw_worker(int(match.group(1)))

        self.run.push(event)

    def _show_page(self, page: str) -> None:
        """Put a page in the tab and send it to the browser."""
        self.run.artifact = page
        if self.build is not None:
            self.build.document = page
        self.run.push({"type": "artifact", "html": page, "sections": len(self.parts)})


def _finish_build(run: Run, error: str) -> None:
    """Close the tab's status line now that the run is over."""
    build = BUILDS.get(run.build_id) if run.build_id else None
    if build is None:
        return
    seconds = time.time() - build.progress.started
    if error:
        build.progress.stopped(seconds)
    else:
        build.progress.built(seconds)
    build.done = True
    build.failed = error
    build.cost = dict(run.cost)
    run.push({"type": "build", "build": build.as_dict(with_document=False)})
    _save_builds()


def _state_event(state: Any, mode: str) -> dict[str, Any]:
    """Flatten one workflow step into something the page can render."""
    artifact = ""
    if mode == "build" and state.subtask_results:
        artifact = assemble_page(
            state.current_task,
            page_fragments(state),
            overlay=True,
            theme=state.context.get("design_theme") or None,
        )

    return {
        "type": "state",
        "messages": len(state.messages),
        "findings": len(state.research_findings),
        "validations": len(state.validation_results),
        "plan": [str(item.get("section") or item.get("task") or "") for item in state.plan],
        "workers": [
            {
                "task": str(item.get("task", "")),
                "chars": len(str(item.get("content", ""))),
                "preview": " ".join(str(item.get("content", "")).split())[:280],
            }
            for item in sorted(state.subtask_results, key=lambda r: r.get("index", 0))
        ],
        "answer": state.final_report or "",
        "artifact": artifact,
        "sections": len(state.subtask_results),
        "errors": [e.get("error", "") for e in state.errors],
    }


async def _drive(run: Run) -> None:
    """Run one turn, streaming progress and recording the reply."""
    from src.config import get_llm
    from src.dashboard.fake_model import scripted_enabled, scripted_llm
    from src.graph.agent import run_agent_stream
    from src.graph.build import run_build_stream
    from src.graph.research import run_research_stream
    from src.graph.team import run_team_stream
    from src.graph.workflow import run_workflow_stream

    meter = start_run(f"{run.mode} · {run.text[:60]}")
    collector = _BuildCollector(run)
    unsubscribe = activity.subscribe(collector.on_event)
    try:
        llm = scripted_llm() if scripted_enabled() else get_llm()
        # fresh=False: a follow-up continues the thread the workflow remembers.
        if run.mode == "agent":
            stream = run_agent_stream(run.query, run.thread_id, llm, fresh=False)
        elif run.mode == "build":
            # A new tab starts a clean thread, so no other page's sections can
            # leak into this one. Editing a tab continues that tab's own page.
            stream = run_build_stream(run.query, run.thread_id, llm, run.subtasks, fresh=run.fresh)
        elif run.mode == "research":
            waves = 2 if run.intensity == "deep" else 1
            stream = run_research_stream(
                run.query, run.thread_id, llm, run.subtasks, waves=waves, fresh=False
            )
        elif run.mode == "pipeline":
            stream = run_workflow_stream(run.query, run.thread_id, llm, fresh=False)
        else:
            stream = run_team_stream(run.query, run.thread_id, llm, run.subtasks, fresh=False)

        async for state in stream:
            event = _state_event(state, run.mode)
            if event["plan"]:
                run.plan = event["plan"]
                collector.announce_plan()
            run.answer = event["answer"] or run.answer
            collector.theme = state.context.get("design_theme") or collector.theme
            run.artifact = event["artifact"] or run.artifact
            run.findings = list(state.research_findings) or run.findings
            collector.on_state(event)

        run.cost = end_run(meter)
        run.done = True
        _finish_build(run, "")
        run.push({"type": "done", "answer": run.answer, "cost": run.cost})

    except Exception as e:
        meter.stop()
        run.error = str(e)
        run.done = True
        _finish_build(run, run.error)
        run.push({"type": "failed", "message": run.error})

    finally:
        unsubscribe()
        findings = [
            {
                "question": str(item.get("question") or ""),
                "content": str(item.get("content") or "")[:400],
                "sources": list(item.get("sources") or [])[:3],
            }
            for item in (run.findings or [])
        ]
        _remember(run.thread_id, run.mode, run.answer, list(run.plan), findings, run.artifact)
        _append(
            run.thread_id,
            {
                "role": "assistant",
                "text": run.answer if run.answer else (run.error or "No answer produced."),
                "mode": run.mode,
                "ts": time.time(),
                "run_id": run.id,
                "sections": len(run.plan),
                "plan": list(run.plan),
                "failed": bool(run.error),
                "cost": run.cost,
            },
        )


def _start_turn(request: ChatRequest) -> dict[str, Any]:
    """Record the user's message and begin the turn."""
    text = request.text.strip()
    if not text:
        return {"error": "say something first"}

    thread_id = request.thread_id.strip() or uuid.uuid4().hex[:12]
    requested = (request.mode or "agent").lower()
    allowed = ("agent", "team", "research", "pipeline", "build")
    mode = requested if requested in allowed else "agent"
    intensity = (request.intensity or "normal").lower()
    if mode == "research" and intensity != "deep":
        intensity = "normal"
    if mode == "build" and intensity != "super":
        intensity = "normal"

    query = text
    build: Build | None = None
    fresh = False
    if mode == "build":
        # A build gets its own tab and its own conversation; editing an open tab
        # is the only way one page's sections reach another request.
        build, fresh = _open_build(request.build_id, text)
        thread_id = build.thread_id
        query = build.request

    _append(thread_id, {"role": "user", "text": text, "mode": mode, "ts": time.time()})

    if mode == "research":
        count = 10 if intensity == "deep" else 4
    elif mode == "build":
        count = 12 if intensity == "super" else 5
    else:
        count = max(1, min(int(request.subtasks), 20))

    run = Run(
        id=uuid.uuid4().hex[:10],
        thread_id=thread_id,
        query=query,
        text=text,
        mode=mode,
        subtasks=count,
        intensity=intensity,
        build_id=build.id if build else "",
        fresh=fresh,
    )
    RUNS[run.id] = run
    while len(RUNS) > MAX_RUNS:
        RUNS.pop(min(RUNS.values(), key=lambda r: r.started).id, None)

    activity.reset()
    from src.computer import reset_session

    reset_session()
    run.push({"type": "started", "query": text, "mode": mode})
    if build is not None:
        run.push({"type": "build", "build": build.as_dict(with_document=False)})
    run.task = asyncio.create_task(_drive(run))
    return {
        "run_id": run.id,
        "thread_id": thread_id,
        "mode": mode,
        "build": build.as_dict(with_document=False) if build else None,
    }


async def _event_stream(run: Run) -> AsyncIterator[str]:
    """Yield one turn's events as server-sent events."""
    sent = 0
    idle = 0.0
    while True:
        while sent < len(run.events):
            yield f"data: {json.dumps(run.events[sent])}\n\n"
            sent += 1
        if run.done and sent >= len(run.events):
            break
        await asyncio.sleep(POLL_SECONDS)
        idle += POLL_SECONDS
        if idle >= KEEPALIVE_SECONDS:
            idle = 0.0
            yield ": keepalive\n\n"
    yield 'data: {"type": "close"}\n\n'


def _load_notes() -> list[dict[str, Any]]:
    """Read the comments pinned to the UI."""
    if not NOTES_FILE.exists():
        return []
    try:
        data = json.loads(NOTES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def create_app() -> FastAPI:
    """Wire the dashboard routes."""
    _load_chats()
    _load_state()
    _load_builds()

    app = FastAPI(title="Multi-agent dashboard", docs_url=None, redoc_url=None)
    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/markdown.js", markdown_script, methods=["GET"])
    app.add_api_route("/api/chat", chat, methods=["POST"])
    app.add_api_route("/api/chat/{thread_id}", history, methods=["GET"])
    app.add_api_route("/api/chat/{thread_id}", delete_thread, methods=["DELETE"])
    app.add_api_route("/api/threads", threads, methods=["GET"])
    app.add_api_route("/api/builds", builds, methods=["GET"])
    app.add_api_route("/api/builds/{build_id}", delete_build, methods=["DELETE"])
    app.add_api_route("/api/stream/{run_id}", stream, methods=["GET"])
    app.add_api_route("/api/cancel/{run_id}", cancel, methods=["POST"])
    app.add_api_route("/api/notes", notes, methods=["GET"])
    app.add_api_route("/api/notes", add_note, methods=["POST"])
    app.add_api_route("/api/notes/{note_id}", delete_note, methods=["DELETE"])
    app.add_api_route("/api/settings", settings_info, methods=["GET"])
    app.add_api_route("/api/settings", settings_update, methods=["POST"])
    app.add_api_route("/api/providers", provider_health, methods=["GET"])
    app.add_api_route("/api/voice", voice_status, methods=["GET"])
    app.add_api_route("/api/voice/transcribe", voice_transcribe, methods=["POST"])
    app.add_api_route("/health", health, methods=["GET"])
    return app


async def index() -> FileResponse:
    """Serve the chat page."""
    return FileResponse(STATIC_DIR / "index.html")


async def markdown_script() -> FileResponse:
    """Serve the vendored markdown renderer the page loads, with no network."""
    return FileResponse(STATIC_DIR / "markdown.js", media_type="text/javascript")


async def chat(request: ChatRequest) -> dict[str, Any]:
    """Start a turn in a thread."""
    return _start_turn(request)


async def history(thread_id: str) -> dict[str, Any]:
    """Return one thread's messages plus whatever it produced."""
    return {
        "thread_id": thread_id,
        "messages": CHATS.get(thread_id, []),
        "state": THREAD_STATE.get(thread_id, {}),
    }


async def delete_thread(thread_id: str) -> dict[str, Any]:
    """Forget a conversation: its visible log and its saved workflow state."""
    CHATS.pop(thread_id, None)
    _save_chats()
    if THREAD_STATE.pop(thread_id, None) is not None:
        _save_state()
    for build_id in [key for key, build in BUILDS.items() if build.thread_id == thread_id]:
        del BUILDS[build_id]
        if build_id in BUILD_ORDER:
            BUILD_ORDER.remove(build_id)
        _save_builds()

    from src.graph.workflow import checkpointer_scope, reset_thread

    try:
        async with checkpointer_scope() as saver:
            await reset_thread(saver, thread_id)
    except Exception as e:  # the log is gone either way
        return {"deleted": True, "state_cleared": False, "note": str(e)}

    return {"deleted": True, "state_cleared": True}


async def threads() -> dict[str, Any]:
    """List conversations, newest first."""
    rows = []
    for thread_id, messages in CHATS.items():
        first = next((m for m in messages if m.get("role") == "user"), None)
        rows.append(
            {
                "thread_id": thread_id,
                "title": (first or {}).get("text", "Untitled")[:60],
                "mode": (first or {}).get("mode", ""),
                "turns": sum(1 for m in messages if m.get("role") == "user"),
                "updated": messages[-1].get("ts", 0) if messages else 0,
            }
        )
    rows.sort(key=lambda row: row["updated"], reverse=True)
    return {"threads": rows}


async def builds() -> dict[str, Any]:
    """Every build's tab, oldest first, each with the page it holds."""
    return {
        "builds": [BUILDS[build_id].as_dict() for build_id in BUILD_ORDER if build_id in BUILDS]
    }


async def delete_build(build_id: str) -> dict[str, Any]:
    """Close one tab and forget the page it held."""
    if BUILDS.pop(build_id, None) is None:
        return {"deleted": False}
    if build_id in BUILD_ORDER:
        BUILD_ORDER.remove(build_id)
    _save_builds()
    return {"deleted": True}


async def stream(run_id: str) -> StreamingResponse:
    """Stream one turn's events."""
    run = RUNS.get(run_id)
    if run is None:
        empty = Run(id=run_id, thread_id="", query="", mode="", subtasks=1, done=True)
        return StreamingResponse(_event_stream(empty), media_type="text/event-stream")
    return StreamingResponse(
        _event_stream(run),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def cancel(run_id: str) -> dict[str, Any]:
    """Stop a turn that is still running."""
    run = RUNS.get(run_id)
    if run is None or run.done:
        return {"cancelled": False}
    if run.task is not None:
        run.task.cancel()
    run.done = True
    run.push({"type": "failed", "message": "Stopped by you."})
    return {"cancelled": True}


async def notes() -> dict[str, Any]:
    """Return the comments pinned to the UI."""
    return {"notes": _load_notes()}


async def add_note(note: NoteRequest) -> dict[str, Any]:
    """Save a comment against one UI element."""
    rows = _load_notes()
    rows.append(
        {
            "id": uuid.uuid4().hex[:8],
            "selector": note.selector,
            "label": note.label,
            "text": note.text,
            "view": note.view,
            "ts": time.time(),
        }
    )
    NOTES_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"saved": True, "count": len(rows)}


async def delete_note(note_id: str) -> dict[str, Any]:
    """Remove one comment."""
    rows = [row for row in _load_notes() if row.get("id") != note_id]
    NOTES_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"deleted": True, "count": len(rows)}


# What the settings sheet may change: attribute, env name, kind, bounds.
EDITABLE_SETTINGS: dict[str, tuple[str, str, str, float | None, float | None]] = {
    "model": ("keypool_model", "KEYPOOL_MODEL", "text", None, None),
    "thinking": ("thinking_level", "THINKING_LEVEL", "choice", None, None),
    "computer": ("computer_use", "COMPUTER_USE", "toggle", None, None),
    "whisper": ("whisper_model", "WHISPER_MODEL", "choice", None, None),
    "whisper_device": ("whisper_device", "WHISPER_DEVICE", "choice", None, None),
    "max_workers": ("max_parallel_agents", "MAX_PARALLEL_AGENTS", "int", 1, 12),
    "attempts": ("max_fallback_retries", "MAX_FALLBACK_RETRIES", "int", 1, 10),
    "backoff": ("llm_retry_backoff", "LLM_RETRY_BACKOFF", "float", 0, 60),
    "tool_attempts": ("tool_retries", "TOOL_RETRIES", "int", 1, 5),
    "timeout": ("llm_timeout_seconds", "LLM_TIMEOUT_SECONDS", "float", 10, 600),
}

MAX_TEXT_SETTING = 80

FIELD_LABELS = {
    "model": "Model",
    "thinking": "Thinking power",
    "computer": "Computer use",
    "whisper": "Speech model",
    "whisper_device": "Speech hardware",
    "max_workers": "Parallel workers",
    "attempts": "Model retries",
    "backoff": "Backoff base (s)",
    "tool_attempts": "Tool retries",
    "timeout": "Model timeout (s)",
}


class SettingsUpdate(BaseModel):
    """Values the settings sheet wants to change."""

    updates: dict[str, Any] = {}


CHOICES = {
    "thinking": ["off", "low", "medium", "high", "max"],
    "computer": ["off", "on"],
    "whisper": ["base", "small", "distil-small.en", "medium", "large-v3"],
    "whisper_device": ["auto", "cuda", "cpu"],
}


def _coerce(key: str, value: Any) -> Any:
    """Validate one incoming setting against its declared kind and bounds."""
    _, _, kind, low, high = EDITABLE_SETTINGS[key]

    if kind in {"choice", "toggle"}:
        choice = str(value).strip().lower()
        allowed = CHOICES.get(key, [])
        if choice not in allowed:
            raise ValueError(f"{FIELD_LABELS[key]} must be one of {', '.join(allowed)}")
        return choice == "on" if kind == "toggle" else choice

    if kind == "text":
        text = str(value).strip()
        if not text or len(text) > MAX_TEXT_SETTING:
            raise ValueError(f"{FIELD_LABELS[key]} must be 1-80 characters")
        return text

    number: float = float(value)
    if low is not None and number < low:
        raise ValueError(f"{FIELD_LABELS[key]} must be at least {low:g}")
    if high is not None and number > high:
        raise ValueError(f"{FIELD_LABELS[key]} must be at most {high:g}")
    return int(number) if kind == "int" else number


def _write_env(changes: dict[str, str]) -> None:
    """Persist settings to .env, leaving every other line untouched."""
    path = PROJECT_ROOT / ".env"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(changes)

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name = stripped.split("=", 1)[0].strip()
        if name in remaining:
            lines[index] = f"{name}={remaining.pop(name)}"

    for name, value in remaining.items():
        lines.append(f"{name}={value}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _settings_payload() -> dict[str, Any]:
    """Everything the settings sheet renders."""
    from src.config import router

    editable = []
    for key, (attribute, _, kind, low, high) in EDITABLE_SETTINGS.items():
        current = getattr(settings, attribute)
        if kind == "toggle":
            current = "on" if current else "off"

        editable.append(
            {
                "key": key,
                "label": FIELD_LABELS[key],
                "value": current,
                "kind": kind,
                "min": low,
                "max": high,
                "choices": CHOICES.get(key, []),
            }
        )

    return {
        "editable": editable,
        "provider": settings.llm_provider,
        "chain": router.provider_names(),
        "model": router.providers[0].model if router.providers else "",
        "base_url": router.providers[0].base_url if router.providers else "",
        "checkpoint_db": settings.checkpoint_db or "(in memory)",
        "max_subtasks": settings.max_parallel_agents,
        "timeout": settings.llm_timeout_seconds,
        "attempts": settings.max_fallback_retries,
        "tool_attempts": settings.tool_retries,
        "backoff": settings.llm_retry_backoff,
        "thinking": settings.thinking_level,
    }


async def settings_info() -> dict[str, Any]:
    """What the dashboard is configured to do."""
    return _settings_payload()


async def settings_update(update: SettingsUpdate) -> dict[str, Any]:
    """Apply and persist setting changes, then report the new state."""
    from src.config import router

    unknown = [key for key in update.updates if key not in EDITABLE_SETTINGS]
    if unknown:
        return {"error": f"not configurable: {', '.join(unknown)}"}

    env_changes: dict[str, str] = {}
    applied: list[str] = []

    try:
        for key, value in update.updates.items():
            attribute, env_name, _, _, _ = EDITABLE_SETTINGS[key]
            coerced = _coerce(key, value)
            setattr(settings, attribute, coerced)
            env_changes[env_name] = str(coerced)
            applied.append(key)
    except (TypeError, ValueError) as e:
        return {"error": str(e)}

    _write_env(env_changes)
    if {"whisper", "whisper_device"} & set(applied):
        from src.voice import forget_model

        forget_model()

    if {"model", "max_workers", "thinking"} & set(applied):
        # Model clients are built once, so rebuild them to pick up the change.
        router.reset()

    payload = _settings_payload()
    payload["saved"] = applied
    return payload


async def voice_status() -> dict[str, Any]:
    """What the local speech engine can do."""
    from src.voice import voice_status as status

    return status()


async def voice_transcribe(request: Request) -> dict[str, Any]:
    """Transcribe recorded audio posted from the browser."""
    from src.voice import VoiceUnavailable, transcribe

    audio = await request.body()
    try:
        text = await transcribe(audio, filename="speech.webm")
    except VoiceUnavailable as e:
        return {"error": str(e)}
    return {"text": text}


async def provider_health() -> dict[str, Any]:
    """Check which providers answer right now."""
    from src.config import router

    results = await router.check_all()
    return {"providers": [{"name": name, "ok": ok} for name, ok in results.items()]}


async def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "healthy", "provider": settings.llm_provider}


def main() -> None:
    """Serve the dashboard."""
    host = settings.dashboard_host
    port = settings.dashboard_port
    # Comparing against the wildcard address to pick a clickable URL, not binding to it.
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host  # nosec B104
    url = f"http://{shown}:{port}"
    print(f"Multi-agent dashboard: {url}  (Ctrl+C to stop)")

    import webbrowser

    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"(could not open a browser automatically: {e})")

    import threading

    def warm_voice() -> None:
        """Load the speech model in the background; the first sentence is slow otherwise."""
        try:
            from src.voice import voice_available, warm_up

            if voice_available():
                warm_up()
        except Exception as e:
            print(f"(speech model not warmed up: {e})")

    threading.Thread(target=warm_voice, name="voice-warmup", daemon=True).start()

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
