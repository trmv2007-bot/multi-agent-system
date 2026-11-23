"""Live on-screen cursors: one per agent, showing what that agent is doing now.

The overlay is a borderless, click-through, always-on-top window that draws a
cursor per agent plus, for whatever is driving this machine, a pointer that
travels to the thing it is about to touch, rings it, and names the action.

Geometry and timing live in ``src.cursor_anim`` with no Tk so they can be tested
without a display; this module only renders. It listens to ``src.activity`` and
is only created when something enables it (``--cursor``).
"""

import ctypes
import gc
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.cursor_anim import (
    GLIDE_MS,
    PointerState,
    Sample,
    clamp,
    label_placement,
    rounded_rect,
)

TRANSPARENT_KEY = "#010203"
POINTER_SIZE = 13
CHIP_HEIGHT = 26
SLOT_SPACING = 46
RIGHT_MARGIN = 40
TOP_MARGIN = 110
FRAME_MS = 33
FLASH_THRESHOLD = 0.5
ESCAPE_KEY = 0x1B
ESCAPE_POLL_FRAMES = 6
VISIBLE_ALPHA = 0.005
RECT_FIELDS = 4

# Fading happens per window, so each soft element is kept below full strength.
RING_PANEL_ALPHA = 0.72
CHIP_HEIGHT_ACTION = 34

# Distinct hues, assigned in the order agents first report activity.
PALETTE = ("#5B8DEF", "#E0763C", "#3FA88A", "#B267C4", "#D9A227", "#4FA3C7", "#C4577A")
BACKGROUND = "#12151C"
TEXT = "#F2F4F8"
MUTED = "#9AA4B2"
CHIP_EDGE = "#2A3240"
POINTER_COLOR = "#5B8DEF"


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


class Panel:
    """A small click-through window whose contents fade as one.

    Tk cannot fade a single canvas item, but it can fade a whole window, so each
    soft element (the ring, the action chip) gets its own little window that is
    positioned once per action and only ever changes alpha afterwards.
    """

    def __init__(self, tk: Any, root: Any) -> None:
        self.window = tk.Toplevel(root)
        self.window.overrideredirect(True)
        self.window.configure(bg=TRANSPARENT_KEY)
        self.window.attributes("-topmost", True)
        try:
            self.window.attributes("-transparentcolor", TRANSPARENT_KEY)
        except tk.TclError:
            pass
        self.window.attributes("-alpha", 0.0)
        self.canvas = tk.Canvas(self.window, bg=TRANSPARENT_KEY, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self._shown = False
        self._alpha = 0.0
        self._rect: tuple[int, int, int, int] | None = None
        self._key: Any = None

    def place(self, x: int, y: int, width: int, height: int) -> None:
        """Move and size the window; nothing else changes per frame."""
        if self._rect == (x, y, width, height):
            return
        self._rect = (x, y, width, height)
        self.canvas.configure(width=width, height=height)
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def paint(self, key: Any, draw: Callable[[Any], None]) -> None:
        """Redraw the contents only when they change; fading is just alpha."""
        if key == self._key:
            return
        self._key = key
        self.canvas.delete("all")
        draw(self.canvas)

    def fade(self, alpha: float) -> None:
        """Set the window's opacity, hiding it completely at zero."""
        alpha = round(clamp(alpha, 0.0, 1.0), 3)
        if alpha == self._alpha:
            return
        self._alpha = alpha
        if alpha <= 0.0:
            if self._shown:
                self.window.withdraw()
                self._shown = False
            return
        self.window.attributes("-alpha", alpha)
        if not self._shown:
            self.window.deiconify()
            self.window.update_idletasks()
            # Layered windows stop compositing if WS_EX_TRANSPARENT is written
            # before the window is on screen, so this waits until it is shown.
            _make_click_through(self.window)
            self._shown = True

    def close(self) -> None:
        """Destroy the window, tolerating a window that is already gone."""
        try:
            self.window.destroy()
        except Exception as e:  # a closing session can lose the window first
            print(f"cursor panel already closed: {e}")


class Overlay:
    """The window: a transparent canvas per agent, and a pointer that aims."""

    def __init__(
        self,
        events: queue.Queue | None = None,
        colour_by_key: bool = True,
        glide_ms: float = GLIDE_MS,
    ) -> None:
        self._events: queue.Queue = events or queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cursors: dict[str, AgentCursor] = {}
        self._pointer = PointerState()
        self._pointer_label = "Agent"
        self._colour_by_key = colour_by_key
        self._glide_ms = glide_ms
        self._root: Any = None
        self._canvas: Any = None
        self._tk: Any = None
        self._ring_panel: Panel | None = None
        self._chip_panel: Panel | None = None
        self._ring_key: Any = None
        self._chip_key: Any = None
        self._screen_w = 0
        self._screen_h = 0
        self._frame = 0
        self._hidden = False

    def on_event(self, event: dict[str, Any]) -> None:
        """Queue a bus event. Runs on the agent's thread, so it never touches Tk."""
        kind = event.get("type")
        if kind in ("action", "action_done", "agent"):
            self._events.put((kind, event))

    def start(self) -> None:
        """Run the window on a daemon thread."""
        self._thread = threading.Thread(target=self._run, name="agent-cursor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the window to close and wait for it to go."""
        self._stop.set()
        self._events.put(("stop", {}))
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
            self._tk = None
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
        self._tk = tk
        self._screen_w = width
        self._screen_h = height
        root.after(FRAME_MS, self._tick)

    # ── event handling ───────────────────────────────────────────────
    def _drain(self, now: float) -> None:
        while True:
            try:
                kind, event = self._events.get_nowait()
            except queue.Empty:
                return

            if kind == "stop":
                self._stop.set()
                return
            if kind == "action":
                self._hidden = False  # escape hid the overlay; a new action revives it
                self._aim(event, now)
            elif kind == "action_done":
                self._pointer.complete(now)
            else:
                self._agent(event)

    def _aim(self, event: dict[str, Any], now: float) -> None:
        """Start one beat: the cursor travels to where the action lands."""
        point = event.get("point")
        self._pointer_label = str(event.get("label") or "Agent")
        self._pointer.aim(
            (float(point[0]), float(point[1])) if point else None,
            str(event.get("text") or ""),
            now,
            bounds=_bounds(event.get("bounds")),
            glide_ms=self._glide_ms,
        )

    def _agent(self, event: dict[str, Any]) -> None:
        """A plain agent status line: its own cursor in a slot on the right."""
        key = str(event.get("key") or "")
        if not key:
            return
        status = str(event.get("status") or "")
        cursor = self._cursors.get(key)
        if cursor is None:
            cursor = AgentCursor(
                key=key,
                label=str(event.get("label") or key),
                color=PALETTE[len(self._cursors) % len(PALETTE)],
                slot=len(self._cursors),
            )
            self._place(cursor)
            self._cursors[key] = cursor

        cursor.label = str(event.get("label") or cursor.label)
        if status != cursor.status:
            cursor.status = status
            cursor.flash = 1.0
            self._place(cursor)

    def _place(self, cursor: AgentCursor) -> None:
        """Give a cursor a slot on the right edge and a little room to drift."""
        base_x = self._screen_w - RIGHT_MARGIN - 260
        base_y = TOP_MARGIN + cursor.slot * SLOT_SPACING
        if cursor.x == 0.0 and cursor.y == 0.0:
            cursor.x, cursor.y = base_x, base_y
        cursor.target_x = base_x + ((cursor.slot * 7) % 22) - 11
        cursor.target_y = min(base_y, self._screen_h - 80)

    # ── drawing ──────────────────────────────────────────────────────
    def _tick(self) -> None:
        if self._stop.is_set():
            self._close()
            return

        if self._frame == 0 and self._root is not None:
            # The window is on screen from this tick on; a click-through style
            # written before that stops a layered window from compositing.
            _make_click_through(self._root)

        now = time.monotonic() * 1000.0
        self._drain(now)
        self._frame += 1
        if self._frame % ESCAPE_POLL_FRAMES == 0 and _escape_pressed():
            self._hidden = True
            self._pointer.hide()

        self._draw(now)

        if self._root is not None:
            self._root.after(FRAME_MS, self._tick)

    def _draw(self, now: float) -> None:
        canvas = self._canvas
        canvas.delete("all")
        sample = self._pointer.at(now)

        if sample.visible and not self._hidden:
            self._draw_pointer(canvas, sample)
        self._draw_ring(sample)
        self._draw_action_chip(sample)

        for cursor in self._cursors.values():
            cursor.x += (cursor.target_x - cursor.x) * 0.18
            cursor.y += (cursor.target_y - cursor.y) * 0.18
            cursor.flash = max(0.0, cursor.flash - 0.04)
            self._draw_cursor(canvas, cursor)

    def _draw_pointer(self, canvas: Any, sample: Sample) -> None:
        """The arrow that travels; its tip sits exactly on the action point."""
        x, y = sample.point
        size = POINTER_SIZE
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
            fill=POINTER_COLOR,
            outline=BACKGROUND,
            width=1,
        )

    def _draw_ring(self, sample: Sample) -> None:
        """A soft rounded outline around the target, never a filled box."""
        if self._hidden or sample.ring is None or sample.ring_alpha <= VISIBLE_ALPHA:
            if self._ring_panel is not None:
                self._ring_panel.fade(0.0)
            return

        panel = self._ring_panel or self._new_panel("ring")
        left, top, right, bottom = (round(v) for v in sample.ring)
        width, height = max(1, right - left), max(1, bottom - top)
        panel.place(left, top, width, height)
        key = (width, height)
        if key != self._ring_key:
            self._ring_key = key
            points = rounded_rect((1.0, 1.0, width - 1.0, height - 1.0), radius=14.0)

            def draw_ring(canvas: Any) -> None:
                canvas.create_polygon(points, fill="", outline=POINTER_COLOR, width=2, smooth=True)

            panel.paint(key, draw_ring)
        panel.fade(sample.ring_alpha * RING_PANEL_ALPHA)

    def _draw_action_chip(self, sample: Sample) -> None:
        """The words beside the ring: what is happening, in a person's terms."""
        if (
            self._hidden
            or sample.ring is None
            or not sample.text
            or sample.label_alpha <= VISIBLE_ALPHA
        ):
            if self._chip_panel is not None:
                self._chip_panel.fade(0.0)
            return

        text = sample.text[:44]
        label = self._pointer_label[:22]
        width = int(clamp(max(len(text) * 6.6, len(label) * 5.4) + 24.0, 132.0, 348.0))
        height = CHIP_HEIGHT_ACTION
        x, y = label_placement(
            sample.ring, (width, height), (float(self._screen_w), float(self._screen_h))
        )

        panel = self._chip_panel or self._new_panel("chip")
        panel.place(int(x), int(y), width, height)
        key = (width, height, text, label)
        if key != self._chip_key:
            self._chip_key = key
            panel.paint(key, lambda c: self._paint_chip(c, width, height, label, text))
        panel.fade(sample.label_alpha)

    @staticmethod
    def _paint_chip(canvas: Any, width: int, height: int, label: str, text: str) -> None:
        """One rounded chip: who is acting, then what it is doing."""
        canvas.create_polygon(
            rounded_rect((0.0, 0.0, float(width), float(height)), radius=9.0),
            fill=BACKGROUND,
            outline=CHIP_EDGE,
            width=1,
        )
        canvas.create_oval(9, height / 2 - 3, 15, height / 2 + 3, fill=POINTER_COLOR, outline="")
        canvas.create_text(21, 5, text=label, anchor="nw", fill=MUTED, font=("Segoe UI", 8))
        canvas.create_text(21, 17, text=text, anchor="nw", fill=TEXT, font=("Segoe UI", 9, "bold"))

    def _new_panel(self, kind: str) -> Panel:
        panel = Panel(self._tk, self._root)
        if kind == "ring":
            self._ring_panel = panel
        else:
            self._chip_panel = panel
        return panel

    def _draw_cursor(self, canvas: Any, cursor: AgentCursor) -> None:
        x, y = cursor.x, cursor.y
        size = POINTER_SIZE

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

    def _close(self) -> None:
        """Tear down every window on the overlay thread."""
        for panel in (self._ring_panel, self._chip_panel):
            if panel is not None:
                panel.close()
        self._ring_panel = None
        self._chip_panel = None
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception as e:  # a session that lost its display first
                print(f"cursor overlay already closed: {e}")
            self._root = None


def _bounds(value: Any) -> tuple[float, float, float, float] | None:
    """Coerce an event's rectangle, or None when it carries nothing usable."""
    try:
        if value is None or len(value) != RECT_FIELDS:
            return None
        left, top, right, bottom = (float(v) for v in value)
        return (left, top, right, bottom)
    except (TypeError, ValueError):
        return None


def _make_click_through(window: Any) -> None:
    """Let every click pass through the overlay, and never take focus."""
    try:
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id()) or window.winfo_id()
        extended = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
        # WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
        flags = extended | 0x80000 | 0x20 | 0x08000000 | 0x80
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, flags)
    except Exception as e:
        print(f"cursor overlay is not click-through: {e}")


def _escape_pressed() -> bool:
    """Whether Escape is down: the hatch that hides the overlay."""
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(ESCAPE_KEY) & 0x8000)
    except Exception:
        return False
