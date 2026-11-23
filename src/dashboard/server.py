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
from src.observability import end_run, start_run

STATIC_DIR = Path(__file__).resolve().parent / "static"
CHAT_FILE = PROJECT_ROOT / "chats.json"
STATE_FILE = PROJECT_ROOT / "thread-state.json"
NOTES_FILE = PROJECT_ROOT / "dashboard-notes.json"

POLL_SECONDS = 0.15
KEEPALIVE_SECONDS = 10.0
MAX_RUNS = 40
MAX_THREADS = 40


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
    task: asyncio.Task[None] | None = None

    def push(self, event: dict[str, Any]) -> None:
        """Record an event for the browser watching this turn."""
        event["t"] = round(time.time() - self.started, 2)
        self.events.append(event)


RUNS: dict[str, Run] = {}
CHATS: dict[str, list[dict[str, Any]]] = {}
# What each conversation produced: the last page, the research, the summary.
THREAD_STATE: dict[str, dict[str, Any]] = {}


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


def _previous_sections(thread_id: str) -> list[str]:
    """The section names of the last page built in this thread."""
    for message in reversed(CHATS.get(thread_id, [])):
        if message.get("plan"):
            return [str(name) for name in message["plan"]]
    return []


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


class NoteRequest(BaseModel):
    """A comment pinned to one element of the page."""

    selector: str
    label: str = ""
    text: str
    view: str = "chat"


# ── driving one turn ─────────────────────────────────────────────────
class _BuildCollector:
    """Rebuilds the page preview every time one section lands."""

    def __init__(self, run: Run) -> None:
        self.run = run
        self.parts: dict[int, str] = {}
        self.theme: dict[str, Any] | None = None
        self.plan_shown = False

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
        self.run.push(
            {
                "type": "artifact",
                "html": assemble_page(self.run.query, blocks, overlay=True),
                "sections": 0,
            }
        )

    def on_event(self, event: dict[str, Any]) -> None:
        """Forward bus events, turning finished pages into previews."""
        if event.get("type") == "page":
            html = str(event.get("html", ""))
            self.run.artifact = html
            self.run.push({"type": "artifact", "html": html, "sections": 0})
            return

        if self.run.mode == "build" and event.get("type") == "section":
            self.parts[int(event["index"])] = str(event["html"])
            ordered = [self.parts[index] for index in sorted(self.parts)]
            self.run.push(
                {
                    "type": "artifact",
                    "html": assemble_page(self.run.query, ordered, overlay=True),
                    "sections": len(ordered),
                }
            )
            return
        self.run.push(event)


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
            # A rebuild keeps the thread's history; per-turn results are scoped,
            # and the prompt is told which sections the page currently has.
            stream = run_build_stream(run.query, run.thread_id, llm, run.subtasks, fresh=False)
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
            run.push(event)

        run.cost = end_run(meter)
        run.done = True
        run.push({"type": "done", "answer": run.answer, "cost": run.cost})

    except Exception as e:
        meter.stop()
        run.error = str(e)
        run.done = True
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

    _append(thread_id, {"role": "user", "text": text, "mode": mode, "ts": time.time()})

    query = text
    if mode == "build":
        previous = _previous_sections(thread_id)
        if previous:
            # A rebuild starts a clean page, so tell the planner what was there.
            query = (
                text
                + chr(10)
                + chr(10)
                + ("The page currently has these sections: " + ", ".join(previous) + ".")
            )

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
    )
    RUNS[run.id] = run
    while len(RUNS) > MAX_RUNS:
        RUNS.pop(min(RUNS.values(), key=lambda r: r.started).id, None)

    activity.reset()
    from src.computer import reset_session

    reset_session()
    run.push({"type": "started", "query": text, "mode": mode})
    run.task = asyncio.create_task(_drive(run))
    return {"run_id": run.id, "thread_id": thread_id, "mode": mode}


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

    app = FastAPI(title="Multi-agent dashboard", docs_url=None, redoc_url=None)
    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/api/chat", chat, methods=["POST"])
    app.add_api_route("/api/chat/{thread_id}", history, methods=["GET"])
    app.add_api_route("/api/chat/{thread_id}", delete_thread, methods=["DELETE"])
    app.add_api_route("/api/threads", threads, methods=["GET"])
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
