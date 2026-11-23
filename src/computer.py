"""Computer use: look at the screen and act on it.

A text model cannot see a screenshot, so the primary sense here is a written
description of the desktop — window titles and, for the focused window, its
controls with their screen coordinates. Screenshots are still captured, so the
person can see what the agent saw, and so a vision-capable model could use them.

Nothing here runs unless ``COMPUTER_USE`` is enabled, and every action counts
against a per-turn budget so a confused agent cannot click forever.
"""

import ctypes
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, settings

CAPTURE_DIR = PROJECT_ROOT / "screenshots"
MAX_DESCRIBED_CONTROLS = 40
CARET_RING_WIDTH = 130

_actions = {"count": 0}

# Where an action lands: the point it touches and the rectangle around it.
Target = tuple[tuple[int, int] | None, tuple[int, int, int, int] | None]

logger = logging.getLogger(__name__)


class _Rect(ctypes.Structure):
    """A Win32 RECT, spelled here so this module imports off Windows too."""

    _fields_ = [
        ("left", ctypes.c_int32),
        ("top", ctypes.c_int32),
        ("right", ctypes.c_int32),
        ("bottom", ctypes.c_int32),
    ]


class _Point(ctypes.Structure):
    """A Win32 POINT."""

    _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_int32)]


class _GuiThreadInfo(ctypes.Structure):
    """The parts of GUITHREADINFO this module reads: the caret and its window."""

    _fields_ = [
        ("cbSize", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("hwndActive", ctypes.c_void_p),
        ("hwndFocus", ctypes.c_void_p),
        ("hwndCapture", ctypes.c_void_p),
        ("hwndMenuOwner", ctypes.c_void_p),
        ("hwndMoveSize", ctypes.c_void_p),
        ("hwndCaret", ctypes.c_void_p),
        ("rcCaret", _Rect),
    ]


class ComputerUnavailable(RuntimeError):
    """Raised when this machine cannot be driven (no desktop, missing library)."""


def reset_session() -> None:
    """Start a fresh action budget. Called once per turn."""
    _actions["count"] = 0


def actions_used() -> int:
    """How many input actions this turn has performed."""
    return _actions["count"]


def _guard() -> Any:
    """Ensure computer use is allowed and possible, then return pyautogui."""
    if not settings.computer_use:
        raise ComputerUnavailable(
            "Computer use is off. Turn it on in Settings (or set COMPUTER_USE=true)."
        )
    try:
        import pyautogui
    except Exception as e:  # pragma: no cover - depends on the machine
        raise ComputerUnavailable(f"pyautogui is not available: {e}") from e
    return pyautogui


def _beat(text: str, run: Callable[[], Any], where: Callable[[], Target]) -> None:
    """Show the action, then do it: aim, settle, act, confirm.

    ``text`` names the action in a person's words, and ``where`` is asked only
    when something is listening, so a run with no overlay pays nothing. Every
    overlay call is cosmetic: a failure here is logged and the action still runs.
    """
    showing = False
    try:
        from src.activity import activity

        if activity.listening:
            point, bounds = where()
            x, y = (float(point[0]), float(point[1])) if point else (None, None)
            activity.action(text, x, y, bounds=bounds)
            showing = True
    except Exception as e:  # the overlay is cosmetic; never break an action for it
        logger.debug("aim failed: %s", e)

    if showing and settings.cursor_settle_ms > 0:
        time.sleep(settings.cursor_settle_ms / 1000.0)
    try:
        run()
    finally:
        _finish(showing)


def _finish(showing: bool) -> None:
    """Tell the overlay the action fired, so the ring can hold and then fade."""
    if not showing:
        return
    try:
        from src.activity import activity

        activity.action_done("Agent")
    except Exception as e:  # the overlay is cosmetic; never break an action for it
        logger.debug("confirm failed: %s", e)


def _pretty_keys(keys: str) -> str:
    """A key chord the way a person writes it: ``ctrl+s`` becomes ``Ctrl+S``."""
    parts = [part.strip() for part in keys.replace("+", " ").split() if part.strip()]
    return "+".join(part.capitalize() for part in parts) or keys


def _foreground_rect() -> tuple[int, int, int, int] | None:
    """The focused window's screen rectangle, read straight from Win32."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        rect = _Rect()
        if not hwnd or not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
    except Exception as e:
        logger.debug("window rect lookup failed: %s", e)
        return None


def _focus_target() -> Target:
    """Where keystrokes will land: the caret's field if the desktop says so, else its window."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        info = _GuiThreadInfo()
        info.cbSize = ctypes.sizeof(_GuiThreadInfo)
        thread = user32.GetWindowThreadProcessId(hwnd, None)
        if thread and user32.GetGUIThreadInfo(thread, ctypes.byref(info)) and info.hwndCaret:
            caret = info.rcCaret
            left, top = int(caret.left), int(caret.top)
            right = max(int(caret.right), left + 3)  # a caret is one pixel wide
            bottom = max(int(caret.bottom), top + 12)
            origin = _Point(0, 0)
            user32.ClientToScreen(info.hwndCaret, ctypes.byref(origin))
            left, right = left + origin.x, right + origin.x
            top, bottom = top + origin.y, bottom + origin.y
            point = ((left + right) // 2, (top + bottom) // 2)
            return point, (
                left - CARET_RING_WIDTH,
                top - 12,
                right + CARET_RING_WIDTH,
                bottom + 12,
            )
    except Exception as e:
        logger.debug("caret lookup failed: %s", e)

    rect = _foreground_rect()
    if rect is None:
        return None, None
    return ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), rect


def _at(x: int | None, y: int | None) -> Target:
    """The point an action was given, or the focused window when it names none."""
    if x is None or y is None:
        return _focus_target()
    return (int(x), int(y)), None


def _spend(what: str) -> None:
    """Count one input action against the budget."""
    if _actions["count"] >= settings.computer_max_actions:
        raise ComputerUnavailable(
            f"Action budget reached ({settings.computer_max_actions}) for this turn. "
            "Stop and report what you did."
        )
    _actions["count"] += 1


def _top_windows() -> list[dict[str, Any]]:
    """Every visible top-level window, front-most first."""
    try:
        from pywinauto import Desktop
    except Exception as e:  # pragma: no cover
        raise ComputerUnavailable(f"pywinauto is not available: {e}") from e

    rows: list[dict[str, Any]] = []
    try:
        for window in Desktop(backend="uia").windows():
            try:
                title = window.window_text().strip()
                rect = window.rectangle()
            except Exception as e:
                logger.debug("window skipped: %s", e)
                title, rect = "", None
            if rect is None:
                continue
            if not title:
                continue
            rows.append(
                {
                    "title": title,
                    "focused": bool(window.is_active()),
                    "rect": [rect.left, rect.top, rect.right, rect.bottom],
                }
            )
    except Exception as e:  # pragma: no cover
        raise ComputerUnavailable(f"cannot read the desktop: {e}") from e

    rows.sort(key=lambda row: not row["focused"])
    return rows


def screen_shot() -> dict[str, Any]:
    """Capture the screen and describe where the cursor is."""
    pyautogui = _guard()
    CAPTURE_DIR.mkdir(exist_ok=True)
    path = CAPTURE_DIR / f"screen-{time.strftime('%H%M%S')}.png"
    try:
        image = pyautogui.screenshot()
    except Exception as e:  # pragma: no cover
        raise ComputerUnavailable(f"cannot capture the screen: {e}") from e

    image.save(path)
    width, height = image.size
    position = pyautogui.position()
    front = next((row["title"] for row in _top_windows() if row["focused"]), "")
    return {
        "saved": str(path),
        "width": width,
        "height": height,
        "cursor": [int(position.x), int(position.y)],
        "focused_window": front,
        "note": "You cannot see this image. Use screen_describe for a written view.",
    }


def _focused_controls(limit: int) -> list[dict[str, Any]]:
    """The focused window's named controls with their screen coordinates."""
    from pywinauto import Desktop

    window = None
    for candidate in Desktop(backend="uia").windows():
        try:
            if candidate.is_active():
                window = candidate
                break
        except Exception as e:
            logger.debug("window activation check failed: %s", e)
    if window is None:
        return []

    controls: list[dict[str, Any]] = []
    for child in window.descendants()[: limit * 3]:
        try:
            name = child.window_text().strip()
            rect = child.rectangle()
            role = child.element_info.control_type
        except Exception as e:
            logger.debug("control skipped: %s", e)
            name = ""
        if not name:
            continue
        controls.append(
            {
                "name": name[:60],
                "type": role,
                "centre": [(rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2],
            }
        )
        if len(controls) >= limit:
            break
    return controls


def screen_describe(limit: int = MAX_DESCRIBED_CONTROLS) -> dict[str, Any]:
    """A written view of the desktop: windows, and the focused window's controls."""
    rows = _top_windows()
    focused = next((row for row in rows if row["focused"]), None)

    controls: list[dict[str, Any]] = []
    if focused:
        try:
            controls = _focused_controls(limit)
        except Exception as e:  # pragma: no cover
            controls = [{"error": f"control listing failed: {e}"}]

    return {
        "windows": rows[:10],
        "focused": focused["title"] if focused else "",
        "controls": controls,
        "hint": (
            "Click a control by its centre coordinates, then describe the screen again "
            "to confirm the result."
        ),
    }


def mouse_click(
    x: int, y: int, button: str = "left", clicks: int = 1, target: str = ""
) -> dict[str, Any]:
    """Click at a point in screen pixels."""
    pyautogui = _guard()
    _spend("click")
    _beat(
        f"Clicking {target}" if target else "Clicking",
        lambda: pyautogui.click(
            x=int(x), y=int(y), button=button, clicks=max(1, min(int(clicks), 2))
        ),
        lambda: _at(x, y),
    )
    return {"clicked": [int(x), int(y)], "button": button, "clicks": clicks}


def mouse_move(x: int, y: int, target: str = "") -> dict[str, Any]:
    """Move the pointer without clicking."""
    pyautogui = _guard()
    _spend("move")
    _beat(
        f"Moving to {target}" if target else "Moving",
        lambda: pyautogui.moveTo(int(x), int(y), duration=0.2),
        lambda: _at(x, y),
    )
    return {"moved_to": [int(x), int(y)]}


def type_text(text: str, target: str = "") -> dict[str, Any]:
    """Type text into whatever has focus (pasted, so any unicode works)."""
    pyautogui = _guard()
    _spend("type")
    method = ["paste"]

    def run() -> None:
        try:
            import pyperclip

            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
        except Exception:
            method[0] = "keystrokes"
            pyautogui.typewrite(text, interval=0.02)

    _beat(f"Typing into {target}" if target else "Typing", run, _focus_target)
    return {"typed": text[:80], "chars": len(text), "method": method[0]}


def press_key(keys: str, target: str = "") -> dict[str, Any]:
    """Press a key or chord, e.g. "enter" or "ctrl+s"."""
    pyautogui = _guard()
    _spend("key")
    parts = [part.strip().lower() for part in keys.replace("+", " ").split() if part.strip()]
    if not parts:
        raise ComputerUnavailable("no key given")

    def run() -> None:
        if len(parts) == 1:
            pyautogui.press(parts[0])
        else:
            pyautogui.hotkey(*parts)

    name = _pretty_keys(keys)
    _beat(f"Pressing {name} in {target}" if target else f"Pressing {name}", run, _focus_target)
    return {"pressed": "+".join(parts)}


def scroll(
    amount: int, x: int | None = None, y: int | None = None, target: str = ""
) -> dict[str, Any]:
    """Scroll the wheel; positive scrolls up."""
    pyautogui = _guard()
    _spend("scroll")
    where = (int(x), int(y)) if x is not None and y is not None else None

    def run() -> None:
        if where is not None:
            pyautogui.moveTo(where[0], where[1], duration=0.15)
        pyautogui.scroll(int(amount))

    direction = "up" if int(amount) >= 0 else "down"
    _beat(f"Scrolling {target}" if target else f"Scrolling {direction}", run, lambda: _at(x, y))
    return {"scrolled": int(amount), "at": [x, y] if where is not None else None}


def screen_size() -> dict[str, int]:
    """Current display size in pixels."""
    pyautogui = _guard()
    size = pyautogui.size()
    return {"width": int(size.width), "height": int(size.height)}


def foreground_geometry() -> dict[str, Any]:
    """The focused window's rectangle, useful for screen coordinates."""
    rows = _top_windows()
    focused = next((row for row in rows if row["focused"]), None)
    if not focused:
        return {"focused": "", "rect": None}
    return {"focused": focused["title"], "rect": focused["rect"]}


def _result(function: Any, **kwargs: Any) -> str:
    """Run one action and always return JSON, never raise at the tool boundary."""
    try:
        return json.dumps(function(**kwargs))
    except ComputerUnavailable as e:
        return json.dumps({"error": str(e)})
    except Exception as e:  # pragma: no cover - depends on the desktop
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


async def tool_screen_shot() -> str:
    """Capture the screen to a file."""
    return _result(screen_shot)


async def tool_screen_describe(limit: int = MAX_DESCRIBED_CONTROLS) -> str:
    """Read the desktop as text."""
    return _result(screen_describe, limit=limit)


async def tool_mouse_click(
    x: int, y: int, button: str = "left", clicks: int = 1, target: str = ""
) -> str:
    """Click at screen pixel coordinates."""
    return _result(mouse_click, x=x, y=y, button=button, clicks=clicks, target=target)


async def tool_type_text(text: str, target: str = "") -> str:
    """Type text into the focused field."""
    return _result(type_text, text=text, target=target)


async def tool_press_key(keys: str, target: str = "") -> str:
    """Press a key or chord."""
    return _result(press_key, keys=keys, target=target)


async def tool_scroll(
    amount: int, x: int | None = None, y: int | None = None, target: str = ""
) -> str:
    """Scroll the wheel at a point."""
    return _result(scroll, amount=amount, x=x, y=y, target=target)


def computer_tools() -> list[Any]:
    """The LangChain tools a computer-using agent gets."""
    from langchain_core.tools import StructuredTool

    return [
        StructuredTool.from_function(
            coroutine=tool_screen_shot,
            name="screen_shot",
            description=(
                "Capture the screen to a file. You cannot see the image; use "
                "screen_describe to read the desktop as text."
            ),
        ),
        StructuredTool.from_function(
            coroutine=tool_screen_describe,
            name="screen_describe",
            description=(
                "Read the desktop: open windows, which one is focused, and the focused "
                "window's named controls with the screen coordinates of each centre."
            ),
        ),
        StructuredTool.from_function(
            coroutine=tool_mouse_click,
            name="mouse_click",
            description=(
                "Click at screen pixel coordinates taken from screen_describe. "
                "Pass target with the control's name in a few words ('Search', "
                "'the Save button') so the person watching sees what is being clicked."
            ),
        ),
        StructuredTool.from_function(
            coroutine=tool_type_text,
            name="type_text",
            description=(
                "Type text into the focused field (pasted, so any characters work). "
                "Pass target naming the field ('the address bar') so the person "
                "watching sees where the text is going."
            ),
        ),
        StructuredTool.from_function(
            coroutine=tool_press_key,
            name="press_key",
            description='Press a key or chord, e.g. "enter", "tab", "ctrl+s", "alt+f4".',
        ),
        StructuredTool.from_function(
            coroutine=tool_scroll,
            name="scroll",
            description=(
                "Scroll the mouse wheel at a point. Positive scrolls up. Pass target "
                "naming what is being scrolled ('the page') when it is not obvious."
            ),
        ),
    ]


def _screen_is_present() -> bool:
    """Whether this session can actually capture the display."""
    try:
        from PIL import ImageGrab

        ImageGrab.grab().close()
        return True
    except Exception:
        try:
            ctypes.windll.user32.GetSystemMetrics(0)  # type: ignore[attr-defined]
            return True
        except Exception:
            return False


def capture_dir() -> Path:
    """Where screenshots land."""
    return CAPTURE_DIR
