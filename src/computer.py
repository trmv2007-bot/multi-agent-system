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
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, settings

CAPTURE_DIR = PROJECT_ROOT / "screenshots"
MAX_DESCRIBED_CONTROLS = 40

_actions = {"count": 0}

logger = logging.getLogger(__name__)


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


def _announce(what: str) -> None:
    """Show where the pointer is now, so the overlay can follow the real mouse."""
    try:
        import pyautogui

        from src.activity import activity

        position = pyautogui.position()
        activity.pointer(position.x, position.y, "Agent", what)
    except Exception as e:  # the overlay is cosmetic; never break an action for it
        logger.debug("pointer announce failed: %s", e)


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


def mouse_click(x: int, y: int, button: str = "left", clicks: int = 1) -> dict[str, Any]:
    """Click at a point in screen pixels."""
    pyautogui = _guard()
    _spend("click")
    _announce("clicking")
    pyautogui.click(x=int(x), y=int(y), button=button, clicks=max(1, min(int(clicks), 2)))
    return {"clicked": [int(x), int(y)], "button": button, "clicks": clicks}


def mouse_move(x: int, y: int) -> dict[str, Any]:
    """Move the pointer without clicking."""
    pyautogui = _guard()
    _spend("move")
    _announce("moving")
    pyautogui.moveTo(int(x), int(y), duration=0.2)
    return {"moved_to": [int(x), int(y)]}


def type_text(text: str) -> dict[str, Any]:
    """Type text into whatever has focus (pasted, so any unicode works)."""
    pyautogui = _guard()
    _spend("type")
    _announce("typing")
    try:
        import pyperclip

        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        method = "paste"
    except Exception:
        pyautogui.typewrite(text, interval=0.02)
        method = "keystrokes"
    return {"typed": text[:80], "chars": len(text), "method": method}


def press_key(keys: str) -> dict[str, Any]:
    """Press a key or chord, e.g. "enter" or "ctrl+s"."""
    pyautogui = _guard()
    _spend("key")
    _announce("pressing")
    parts = [part.strip().lower() for part in keys.replace("+", " ").split() if part.strip()]
    if not parts:
        raise ComputerUnavailable("no key given")
    if len(parts) == 1:
        pyautogui.press(parts[0])
    else:
        pyautogui.hotkey(*parts)
    return {"pressed": "+".join(parts)}


def scroll(amount: int, x: int | None = None, y: int | None = None) -> dict[str, Any]:
    """Scroll the wheel; positive scrolls up."""
    pyautogui = _guard()
    _spend("scroll")
    _announce("scrolling")
    if x is not None and y is not None:
        pyautogui.moveTo(int(x), int(y), duration=0.15)
    pyautogui.scroll(int(amount))
    return {"scrolled": int(amount), "at": [x, y] if x is not None else None}


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


async def tool_mouse_click(x: int, y: int, button: str = "left", clicks: int = 1) -> str:
    """Click at screen pixel coordinates."""
    return _result(mouse_click, x=x, y=y, button=button, clicks=clicks)


async def tool_type_text(text: str) -> str:
    """Type text into the focused field."""
    return _result(type_text, text=text)


async def tool_press_key(keys: str) -> str:
    """Press a key or chord."""
    return _result(press_key, keys=keys)


async def tool_scroll(amount: int, x: int | None = None, y: int | None = None) -> str:
    """Scroll the wheel at a point."""
    return _result(scroll, amount=amount, x=x, y=y)


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
            description="Click at screen pixel coordinates taken from screen_describe.",
        ),
        StructuredTool.from_function(
            coroutine=tool_type_text,
            name="type_text",
            description="Type text into the focused field (pasted, so any characters work).",
        ),
        StructuredTool.from_function(
            coroutine=tool_press_key,
            name="press_key",
            description='Press a key or chord, e.g. "enter", "tab", "ctrl+s", "alt+f4".',
        ),
        StructuredTool.from_function(
            coroutine=tool_scroll,
            name="scroll",
            description="Scroll the mouse wheel at a point. Positive scrolls up.",
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
