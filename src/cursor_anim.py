"""Pure timing and geometry for the live cursor overlay.

Nothing here imports Tk, touches the screen, or reads the clock: every value is
passed in, so the glide, the ring, the label and the fade can be tested with no
display. ``src.overlay`` renders what these functions and classes decide.

The beat for one action is: the cursor glides to the target, the ring pulses
once as it arrives, the action fires, the ring fades out, and the label goes a
couple of seconds later.
"""

import math
from dataclasses import dataclass

Point = tuple[float, float]
Bounds = tuple[float, float, float, float]  # left, top, right, bottom in screen pixels

# Timing, in milliseconds.
GLIDE_MS = 200.0
RING_FADE_IN_MS = 130.0
RING_PULSE_MS = 260.0
RING_HOLD_MS = 220.0
RING_FADE_OUT_MS = 280.0
LABEL_FADE_IN_MS = 90.0
LABEL_HOLD_MS = 2000.0
LABEL_FADE_OUT_MS = 220.0
IDLE_HIDE_MS = 6000.0

# How strong each part gets: readable, never opaque.
RING_PEAK = 0.9
RING_REST = 0.5
LABEL_ALPHA = 0.95

# Geometry.
RING_PADDING = 9.0
MIN_RING = 22.0
LABEL_GAP = 12.0
LABEL_MARGIN = 10.0


def clamp(value: float, low: float, high: float) -> float:
    """Keep a value inside a range."""
    return max(low, min(high, value))


def ease_out(t: float) -> float:
    """Cubic ease-out over 0..1: a quick start and a soft landing."""
    return 1.0 - (1.0 - clamp(t, 0.0, 1.0)) ** 3


def ramp(elapsed: float, start: float, length: float) -> float:
    """0 before a window, then 0 to 1 across it, then 1."""
    if length <= 0.0:
        return 1.0 if elapsed >= start else 0.0
    return clamp((elapsed - start) / length, 0.0, 1.0)


def lerp(start: Point, end: Point, t: float) -> Point:
    """A point part-way from start to end on an ease-out curve."""
    eased = ease_out(t)
    return (
        start[0] + (end[0] - start[0]) * eased,
        start[1] + (end[1] - start[1]) * eased,
    )


def ring_bounds(bounds: Bounds, padding: float = RING_PADDING) -> Bounds:
    """The ring for a rectangle: padded, ordered, and never smaller than MIN_RING."""
    left, top, right, bottom = bounds
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    left, top, right, bottom = left - padding, top - padding, right + padding, bottom + padding
    mid_x, mid_y = (left + right) / 2.0, (top + bottom) / 2.0
    if right - left < MIN_RING:
        left, right = mid_x - MIN_RING / 2.0, mid_x + MIN_RING / 2.0
    if bottom - top < MIN_RING:
        top, bottom = mid_y - MIN_RING / 2.0, mid_y + MIN_RING / 2.0
    return (left, top, right, bottom)


def ring_around(point: Point, padding: float = RING_PADDING) -> Bounds:
    """The ring for a bare coordinate: a padded box centred on the point."""
    return ring_bounds((point[0], point[1], point[0], point[1]), padding)


def ring_contains(bounds: Bounds, ring: Bounds) -> bool:
    """Whether a ring surrounds its target."""
    left, top, right, bottom = ring_bounds(bounds, 0.0)
    return left >= ring[0] and top >= ring[1] and right <= ring[2] and bottom <= ring[3]


def centre(bounds: Bounds) -> Point:
    """The middle of a rectangle."""
    return ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)


def label_placement(
    ring: Bounds,
    size: tuple[float, float],
    screen: tuple[float, float],
    gap: float = LABEL_GAP,
    margin: float = LABEL_MARGIN,
) -> Point:
    """Top-left corner for the action chip: beside the ring, always on screen."""
    width, height = size
    x = ring[2] + gap
    if x + width > screen[0] - margin:
        x = ring[0] - gap - width
    x = clamp(x, margin, max(margin, screen[0] - width - margin))
    y = ring[3] + gap
    if y + height > screen[1] - margin:
        y = ring[1] - gap - height
    y = clamp(y, margin, max(margin, screen[1] - height - margin))
    return (x, y)


def rounded_rect(bounds: Bounds, radius: float = 12.0, steps: int = 4) -> list[float]:
    """Canvas polygon coordinates (flat x, y pairs) for a rounded rectangle."""
    left, top, right, bottom = bounds
    radius = max(0.0, min(radius, (right - left) / 2.0, (bottom - top) / 2.0))
    coords: list[float] = []
    corners = (
        (right - radius, top + radius, -90.0),
        (right - radius, bottom - radius, 0.0),
        (left + radius, bottom - radius, 90.0),
        (left + radius, top + radius, 180.0),
    )
    for corner_x, corner_y, start in corners:
        for step in range(steps + 1):
            angle = math.radians(start + 90.0 * step / steps)
            coords.extend(
                (corner_x + radius * math.cos(angle), corner_y + radius * math.sin(angle))
            )
    return coords


@dataclass(slots=True)
class Sample:
    """What one pointer looks like at one instant."""

    point: Point
    ring: Bounds | None = None
    ring_alpha: float = 0.0
    text: str = ""
    label_alpha: float = 0.0
    visible: bool = False

    @property
    def active(self) -> bool:
        """Whether anything at all should be on screen."""
        return self.visible and (self.ring_alpha > 0.0 or self.label_alpha > 0.0)


class PointerState:
    """One agent's pointer: where it is, what it aims at, what the eye should see.

    Times are milliseconds from any monotonic clock; the caller stamps them, so
    a test can drive a whole beat without sleeping.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self._here: Point = (x, y)
        self._from: Point = (x, y)
        self._target: Point | None = None
        self._ring: Bounds | None = None
        self._text = ""
        self._aimed = 0.0
        self._glide = GLIDE_MS
        self._done: float | None = None
        self._seen = 0.0
        self._visible = False
        self._placed = False

    # ── the beat ─────────────────────────────────────────────────────
    def aim(
        self,
        point: Point | None,
        text: str,
        now: float,
        bounds: Bounds | None = None,
        glide_ms: float = GLIDE_MS,
    ) -> None:
        """Start a beat: travel to point, ring bounds, and name the action in text.

        The first action of a session appears on its target rather than swooshing
        in from the corner. With no point there is nothing to travel to and
        nothing to ring; the label still shows, so an action with no known
        target still reads.
        """
        first = not self._placed
        if first and point is not None:
            self._here = point
        self._from = self._here
        self._target = point
        if bounds is not None:
            self._ring = ring_bounds(bounds)
        elif point is not None:
            self._ring = ring_around(point)
        else:
            self._ring = None
        if point is None and self._ring is not None:
            self._target = centre(self._ring)
        self._text = text
        self._aimed = now
        self._glide = 0.0 if first else max(0.0, glide_ms)
        self._placed = True
        self._done = None
        self._seen = now
        self._visible = True

    def complete(self, now: float) -> None:
        """The action fired: the ring holds briefly, then fades."""
        if self._visible:
            self._done = now
            self._seen = now

    def hide(self) -> None:
        """Drop everything: no ring, no label, no pointer."""
        self._visible = False
        self._ring = None
        self._text = ""
        self._target = None

    # ── what to draw ─────────────────────────────────────────────────
    def at(self, now: float) -> Sample:
        """The pointer at this instant."""
        point = self._position(now)
        if not self._visible or self.stale(now):
            return Sample(point)
        return Sample(
            point, self._ring, self.ring_alpha(now), self._text, self.label_alpha(now), True
        )

    def arrives_at(self) -> float:
        """When the cursor reaches its target."""
        return self._aimed + self._glide

    def stale(self, now: float) -> bool:
        """True once nothing has happened for long enough to drop the pointer."""
        return now - self._seen > IDLE_HIDE_MS

    def ring_alpha(self, now: float) -> float:
        """Invisible in flight, one pulse on arrival, then a fade after the action."""
        arrive = self.arrives_at()
        if not self._visible or self._ring is None or now < arrive:
            return 0.0
        settled = arrive + RING_FADE_IN_MS + RING_PULSE_MS
        if self._done is None:
            # The action has not fired yet: hold at rest until it does.
            return self._pulse(now - arrive) if now < settled else RING_REST
        hold_until = max(settled, self._done + RING_HOLD_MS)
        if now < hold_until:
            return self._pulse(now - arrive)
        return RING_REST * (1.0 - ramp(now - hold_until, 0.0, RING_FADE_OUT_MS))

    def label_alpha(self, now: float) -> float:
        """The action chip fades in on arrival and leaves after a quiet spell."""
        arrive = self.arrives_at()
        if not self._visible or not self._text or now < arrive:
            return 0.0
        shown = LABEL_ALPHA * ramp(now - arrive, 0.0, LABEL_FADE_IN_MS)
        end = max(arrive, self._seen) + LABEL_HOLD_MS
        if now <= end:
            return shown
        return shown * (1.0 - ramp(now - end, 0.0, LABEL_FADE_OUT_MS))

    # ── internals ────────────────────────────────────────────────────
    def _position(self, now: float) -> Point:
        """Where the cursor tip is now, easing toward the target."""
        if self._target is not None and self._glide > 0.0:
            self._here = lerp(self._from, self._target, ramp(now - self._aimed, 0.0, self._glide))
        elif self._target is not None:
            self._here = self._target
        return self._here

    @staticmethod
    def _pulse(since_arrival: float) -> float:
        """Fade in to the peak, then settle to the resting alpha: one pulse."""
        if since_arrival < RING_FADE_IN_MS:
            return RING_PEAK * ramp(since_arrival, 0.0, RING_FADE_IN_MS)
        return RING_PEAK - (RING_PEAK - RING_REST) * ramp(
            since_arrival - RING_FADE_IN_MS, 0.0, RING_PULSE_MS
        )
