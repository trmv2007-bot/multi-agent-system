"""Live on-screen cursors: one per agent, showing what that agent is doing now.

The overlay is a borderless, click-through, always-on-top window that draws a
cursor and a status chip for every agent that reports activity. It listens to
``src.activity`` and is only created when something enables it (``--cursor``).
"""

import gc
import queue
import threading
from dataclasses import dataclass, field
from typing import Any

TRANSPARENT_KEY = "#010203"
POINTER_SIZE = 13
CHIP_HEIGHT = 26
SLOT_SPACING = 46
RIGHT_MARGIN = 40
TOP_MARGIN = 110
FRAME_MS = 33
FLASH_THRESHOLD = 0.5

# Distinct hues, assigned in the order agents first report activity.
PALETTE = ("#5B8DEF", "#E0763C", "#3FA88A", "#B267C4", "#D9A227", "#4FA3C7", "#C4577A")
BACKGROUND = "#12151C"
TEXT = "#F2F4F8"
MUTED = "#9AA4B2"


@dataclass
class AgentCursor:
    """One agent's cursor and status chip."""

    key: str
    label: str
    color: str
    status: str = "starting"
    slot: int = 0
    x: float = 0.0
    y: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    flash: float = 0.0
    seen: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class Overlay:
    """The window: a transparent canvas with one cursor per agent."""

    def __init__(self, events: queue.Queue | None = None, colour_by_key: bool = True) -> None:
        self._events: queue.Queue = events or queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cursors: dict[str, AgentCursor] = {}
        self._colour_by_key = colour_by_key
        self._root: Any = None
        self._canvas: Any = None
        self._pointer: tuple[float, float] | None = None
        self._frame = 0

    def on_event(self, event: dict[str, Any]) -> None:
        """Translate a bus event into a cursor update."""
        kind = event.get("type")

        if kind == "pointer":
            # The agent is acting on this machine: sit on the real pointer.
            self._events.put(
                ("pointer", event.get("label", "Agent"), event.get("status", ""), "pointer")
            )
            self._pointer = (float(event.get("x", 0)), float(event.get("y", 0)))
            return

        if kind == "agent":
            self._events.put((event["key"], event["label"], event["status"], "update"))

    def start(self) -> None:
        """Run the window on a daemon thread."""
        self._thread = threading.Thread(target=self._run, name="agent-cursor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the window to close and wait for it to go."""
        self._stop.set()
        self._events.put(("/control", "", "", "stop"))
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            self._build()
            if self._root is not None:
                self._root.mainloop()
        except Exception as e:  # pragma: no cover - depends on the desktop session
            # A machine without a usable display simply gets no overlay.
            print(f"cursor overlay unavailable: {e}")
        finally:
            # Release the Tcl interpreter on the thread that created it.
            self._root = None
            self._canvas = None
            gc.collect()

    def _build(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        root.title("agent cursors")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.96)
        try:
            root.attributes("-transparentcolor", TRANSPARENT_KEY)
        except tk.TclError:
            pass

        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.geometry(f"{width}x{height}+0+0")
        root.configure(bg=TRANSPARENT_KEY)

        canvas = tk.Canvas(
            root,
            width=width,
            height=height,
            bg=TRANSPARENT_KEY,
            highlightthickness=0,
            bd=0,
        )
        canvas.pack(fill="both", expand=True)

        self._root = root
        self._canvas = canvas
        self._screen_w = width
        self._screen_h = height
        self._make_click_through(root)
        root.after(FRAME_MS, self._tick)

    @staticmethod
    def _make_click_through(root: Any) -> None:
        """Let every click pass through the overlay to the app underneath."""
        try:
            import ctypes

            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            extended = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            # WS_EX_LAYERED | WS_EX_TRANSPARENT
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, extended | 0x80000 | 0x20)
        except Exception as e:
            print(f"cursor overlay is not click-through: {e}")

    # ── event handling ───────────────────────────────────────────────
    def _drain(self) -> None:
        while True:
            try:
                key, label, status, kind = self._events.get_nowait()
            except queue.Empty:
                return

            if kind == "stop" or key == "/control":
                self._stop.set()
                return
            if kind == "remove":
                self._cursors.pop(key, None)
                continue
            if kind == "pointer":
                pointer = self._cursors.get("pointer")
                if pointer is None:
                    pointer = AgentCursor(key="pointer", label=label or "Agent", color=PALETTE[0])
                    self._cursors["pointer"] = pointer
                pointer.label = label or "Agent"
                pointer.status = status
                continue

            cursor = self._cursors.get(key)
            if cursor is None:
                cursor = AgentCursor(
                    key=key,
                    label=label or key,
                    color=PALETTE[len(self._cursors) % len(PALETTE)],
                    slot=len(self._cursors),
                )
                self._place(cursor)
                self._cursors[key] = cursor

            if status != cursor.status:
                cursor.status = status
                cursor.flash = 1.0
                self._place(cursor)

    def _place(self, cursor: AgentCursor) -> None:
        """Give a cursor a slot on the right edge and a little room to drift."""
        if cursor.key == "pointer":
            return
        base_x = self._screen_w - RIGHT_MARGIN - 260
        base_y = TOP_MARGIN + cursor.slot * SLOT_SPACING
        if cursor.x == 0.0 and cursor.y == 0.0:
            cursor.x, cursor.y = base_x, base_y
        cursor.target_x = base_x + ((cursor.slot * 7) % 22) - 11
        cursor.target_y = min(base_y, self._screen_h - 80)

    # ── drawing ──────────────────────────────────────────────────────
    def _tick(self) -> None:
        if self._stop.is_set():
            if self._root is not None:
                self._root.destroy()
                # Drop the reference in the UI thread that created it.
                self._root = None
            return

        self._drain()
        self._frame += 1
        canvas = self._canvas
        canvas.delete("all")

        for cursor in self._cursors.values():
            if cursor.key == "pointer" and self._pointer is not None:
                # Follow the real mouse, quickly enough to read as one pointer.
                cursor.target_x, cursor.target_y = self._pointer
                cursor.x += (cursor.target_x - cursor.x) * 0.55
                cursor.y += (cursor.target_y - cursor.y) * 0.55
                cursor.flash = max(0.0, cursor.flash - 0.04)
                self._draw_cursor(canvas, cursor)
                continue
            cursor.x += (cursor.target_x - cursor.x) * 0.18
            cursor.y += (cursor.target_y - cursor.y) * 0.18
            cursor.flash = max(0.0, cursor.flash - 0.04)
            self._draw_cursor(canvas, cursor)

        if self._root is not None:
            self._root.after(FRAME_MS, self._tick)

    def _draw_cursor(self, canvas: Any, cursor: AgentCursor) -> None:
        x, y = cursor.x, cursor.y
        size = POINTER_SIZE

        # pointer
        canvas.create_polygon(
            x,
            y,
            x,
            y + size,
            x + size * 0.34,
            y + size * 0.72,
            x + size * 0.55,
            y + size,
            x + size * 0.78,
            y + size * 0.9,
            x + size * 0.6,
            y + size * 0.64,
            x + size,
            y + size * 0.6,
            fill=cursor.color,
            outline=BACKGROUND,
            width=1,
        )

        # status chip
        chip_x = x + size + 6
        chip_w = 244
        chip_y = y + 2
        outline = TEXT if cursor.flash > FLASH_THRESHOLD else cursor.color
        canvas.create_rectangle(
            chip_x,
            chip_y,
            chip_x + chip_w,
            chip_y + CHIP_HEIGHT,
            fill=BACKGROUND,
            outline=outline,
            width=1,
        )
        canvas.create_oval(
            chip_x + 8,
            chip_y + CHIP_HEIGHT / 2 - 3,
            chip_x + 14,
            chip_y + CHIP_HEIGHT / 2 + 3,
            fill=cursor.color,
            outline="",
        )
        canvas.create_text(
            chip_x + 20,
            chip_y + 8,
            text=cursor.label[:24],
            anchor="nw",
            fill=TEXT,
            font=("Segoe UI", 9, "bold"),
        )
        canvas.create_text(
            chip_x + 20,
            chip_y + 19,
            text=self._status_text(cursor),
            anchor="nw",
            fill=MUTED,
            font=("Segoe UI", 8),
        )

    def _status_text(self, cursor: AgentCursor) -> str:
        """Status line, with a small pulse while the agent is working."""
        text = cursor.status[:38]
        if cursor.status in ("thinking", "writing", "planning"):
            dots = "." * (1 + (self._frame // 8) % 3)
            text = f"{text}{dots}"
        return text
