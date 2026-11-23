"""Tests for the live cursor: its glide, ring, label, and the actions behind it."""

import time
from types import SimpleNamespace

import pytest

from src import computer
from src.activity import activity
from src.config import settings
from src.cursor_anim import (
    IDLE_HIDE_MS,
    LABEL_ALPHA,
    LABEL_FADE_OUT_MS,
    LABEL_HOLD_MS,
    RING_REST,
    PointerState,
    ease_out,
    label_placement,
    ring_around,
    ring_bounds,
    ring_contains,
    rounded_rect,
)
from src.overlay import Overlay


def _no_display() -> None:
    raise RuntimeError("no display in this session")


def _placed(x: float, y: float) -> PointerState:
    """A pointer that has already been drawn once, so the next aim glides."""
    pointer = PointerState()
    pointer.aim((x, y), "", now=0.0, glide_ms=0.0)
    return pointer


class TestGlide:
    """The cursor travels to where the action lands, then stays there."""

    def test_interpolates_toward_the_target_and_arrives(self):
        pointer = _placed(0.0, 0.0)
        pointer.aim((400.0, 200.0), "Clicking Search", now=100.0, glide_ms=200.0)

        assert pointer.at(100.0).point == (0.0, 0.0)

        halfway = pointer.at(200.0).point
        assert 0.0 < halfway[0] < 400.0
        assert 0.0 < halfway[1] < 200.0
        assert halfway[0] > 200.0  # ease-out is past the midpoint at half the time

        assert pointer.at(300.0).point == (400.0, 200.0)
        assert pointer.at(900.0).point == (400.0, 200.0)

    def test_travel_only_ever_moves_toward_the_target(self):
        pointer = _placed(10.0, 10.0)
        pointer.aim((300.0, 300.0), "Clicking", now=0.0, glide_ms=200.0)
        xs = [pointer.at(t).point[0] for t in range(0, 260, 20)]
        assert xs == sorted(xs)

    def test_the_first_action_appears_on_its_target(self):
        pointer = PointerState()
        pointer.aim((300.0, 200.0), "Clicking", now=0.0, glide_ms=200.0)
        assert pointer.at(0.0).point == (300.0, 200.0)

    def test_a_zero_glide_lands_immediately(self):
        pointer = _placed(5.0, 5.0)
        pointer.aim((60.0, 70.0), "Clicking", now=1000.0, glide_ms=0.0)
        assert pointer.at(1000.0).point == (60.0, 70.0)

    def test_ease_out_is_a_clamped_curve(self):
        assert ease_out(0.0) == 0.0
        assert ease_out(1.0) == 1.0
        assert ease_out(0.5) > 0.5
        assert ease_out(-3.0) == 0.0
        assert ease_out(9.0) == 1.0


class TestRing:
    """The ring surrounds what is being worked on and fades cleanly."""

    def test_ring_bounds_contain_the_target(self):
        targets = [(10, 20, 110, 60), (300, 300, 300, 300), (50, 50, 10, 10), (-5, -8, 40, 9)]
        for target in targets:
            assert ring_contains(target, ring_bounds(target))

    def test_ring_around_a_point_is_a_readable_box(self):
        left, top, right, bottom = ring_around((100.0, 100.0))
        assert left < 100.0 < right
        assert top < 100.0 < bottom
        assert right - left >= 22.0
        assert bottom - top >= 22.0

    def test_ring_is_invisible_until_the_cursor_arrives(self):
        pointer = _placed(0.0, 0.0)
        pointer.aim((200.0, 200.0), "Clicking", now=0.0, glide_ms=200.0)
        assert pointer.ring_alpha(100.0) == 0.0
        assert pointer.ring_alpha(200.0) == 0.0  # arrival: the fade-in starts here
        assert pointer.ring_alpha(240.0) > 0.0

    def test_ring_pulses_once_then_fades(self):
        pointer = _placed(0.0, 0.0)
        pointer.aim((200.0, 200.0), "Clicking", now=0.0, glide_ms=200.0)
        pointer.complete(260.0)
        peak = max(pointer.ring_alpha(t) for t in range(200, 320, 10))
        assert peak > RING_REST
        assert pointer.ring_alpha(700.0) > 0.0  # still fading out here
        assert pointer.ring_alpha(2600.0) == 0.0

    def test_ring_stays_up_while_the_action_runs(self):
        pointer = _placed(0.0, 0.0)
        pointer.aim((200.0, 200.0), "Typing", now=0.0, glide_ms=200.0)
        assert pointer.ring_alpha(1500.0) > 0.0  # never completed: still acting

    def test_ring_geometry_for_a_rounded_outline(self):
        coords = rounded_rect((0.0, 0.0, 100.0, 60.0), radius=10.0)
        xs, ys = coords[0::2], coords[1::2]
        assert len(xs) == len(ys) == 20
        assert min(xs) >= -0.001
        assert max(xs) <= 100.001
        assert min(ys) >= -0.001
        assert max(ys) <= 60.001

    def test_label_placement_keeps_the_chip_on_screen(self):
        screen = (1920.0, 1080.0)
        size = (220.0, 34.0)
        for ring in [(10, 10, 60, 40), (1760, 980, 1900, 1070), (900, 500, 940, 540)]:
            x, y = label_placement(ring, size, screen)
            assert 0.0 <= x <= screen[0] - size[0]
            assert 0.0 <= y <= screen[1] - size[1]


class TestLabel:
    """The action chip reads, then leaves when nothing else happens."""

    def test_label_fades_in_after_the_cursor_arrives(self):
        pointer = _placed(0.0, 0.0)
        pointer.aim((200.0, 200.0), "Clicking Search", now=0.0, glide_ms=200.0)
        assert pointer.label_alpha(150.0) == 0.0
        assert pointer.label_alpha(500.0) == pytest.approx(LABEL_ALPHA)

    def test_stale_label_expires(self):
        pointer = _placed(0.0, 0.0)
        pointer.aim((100.0, 100.0), "Clicking Search", now=0.0, glide_ms=200.0)
        pointer.complete(220.0)
        assert pointer.label_alpha(1000.0) > 0.9
        gone = 220.0 + LABEL_HOLD_MS + LABEL_FADE_OUT_MS + 50.0
        assert pointer.label_alpha(gone) == 0.0
        assert pointer.at(gone).active is False

    def test_pointer_goes_away_after_a_quiet_spell(self):
        pointer = PointerState()
        pointer.aim((10.0, 10.0), "Clicking", now=0.0)
        pointer.complete(200.0)
        assert pointer.stale(IDLE_HIDE_MS + 100.0) is False
        assert pointer.stale(IDLE_HIDE_MS + 300.0) is True
        assert pointer.at(IDLE_HIDE_MS + 300.0).visible is False


class TestMissingTarget:
    """An action with no known target still reads, and hide clears everything."""

    def test_an_action_with_no_target_still_names_itself(self):
        pointer = PointerState(50.0, 60.0)
        pointer.aim(None, "Typing", now=0.0, glide_ms=200.0)
        sample = pointer.at(400.0)
        assert sample.point == (50.0, 60.0)
        assert sample.ring is None
        assert sample.ring_alpha == 0.0
        assert sample.text == "Typing"
        assert sample.label_alpha > 0.9
        assert sample.visible is True

    def test_an_empty_text_draws_no_chip(self):
        pointer = PointerState()
        pointer.aim((80.0, 90.0), "", now=0.0, glide_ms=200.0)
        sample = pointer.at(400.0)
        assert sample.ring_alpha > 0.0
        assert sample.label_alpha == 0.0

    def test_hide_drops_everything_at_once(self):
        pointer = PointerState()
        pointer.aim((10.0, 10.0), "Clicking", now=0.0)
        pointer.complete(200.0)
        pointer.hide()
        sample = pointer.at(300.0)
        assert sample.visible is False
        assert sample.ring is None
        assert sample.label_alpha == 0.0


class TestActivityEvents:
    """The bus carries the target, and a broken listener never breaks a publisher."""

    def test_an_action_event_carries_its_target(self):
        seen: list[dict[str, object]] = []
        unsubscribe = activity.subscribe(seen.append)
        try:
            activity.action("Clicking Search", 120.0, 240.0, bounds=(100, 200, 140, 260))
            activity.action_done()
        finally:
            unsubscribe()

        assert [event["type"] for event in seen] == ["action", "action_done"]
        assert seen[0]["text"] == "Clicking Search"
        assert seen[0]["point"] == [120.0, 240.0]
        assert seen[0]["bounds"] == [100.0, 200.0, 140.0, 260.0]

    def test_an_action_without_coordinates_is_still_published(self):
        seen: list[dict[str, object]] = []
        unsubscribe = activity.subscribe(seen.append)
        try:
            activity.action("Typing")
        finally:
            unsubscribe()
        assert seen[0]["point"] is None
        assert seen[0]["bounds"] is None

    def test_a_broken_listener_never_breaks_the_publisher(self):
        def broken(event: dict[str, object]) -> None:
            raise RuntimeError("boom")

        unsubscribe = activity.subscribe(broken)
        try:
            activity.action("Clicking", 1.0, 2.0)
            activity.action_done()
        finally:
            unsubscribe()


class TestOverlayWithoutADisplay:
    """A session with no usable display loses the overlay, not the run."""

    def test_a_missing_display_degrades_to_no_overlay(self, monkeypatch):
        overlay = Overlay()
        monkeypatch.setattr(overlay, "_build", _no_display)
        overlay.start()
        overlay.on_event({"type": "action", "text": "Clicking", "point": [10.0, 10.0]})
        overlay.on_event({"type": "action_done"})
        overlay.on_event({"type": "agent", "key": "a", "label": "A", "status": "thinking"})
        overlay.stop()

    def test_events_are_only_queued_on_the_agent_thread(self):
        overlay = Overlay()
        overlay.on_event({"type": "action", "text": "Typing", "point": [1.0, 2.0]})
        overlay.on_event({"type": "nonsense"})
        assert overlay._events.qsize() == 1


class TestActionsWithoutAnOverlay:
    """With nothing listening, an action costs no delay and behaves as before."""

    def test_an_action_runs_immediately_when_nobody_watches(self, monkeypatch):
        monkeypatch.setattr(settings, "cursor_settle_ms", 400)
        ran: list[bool] = []
        started = time.monotonic()
        computer._beat("Clicking", lambda: ran.append(True), lambda: ((1, 2), None))
        assert ran == [True]
        assert time.monotonic() - started < 0.2

    def test_the_settle_delay_waits_while_the_overlay_is_on(self, monkeypatch):
        slept: list[float] = []
        monkeypatch.setattr(computer, "time", SimpleNamespace(sleep=slept.append))
        monkeypatch.setattr(settings, "cursor_settle_ms", 150)
        unsubscribe = activity.subscribe(lambda event: None)
        try:
            computer._beat("Clicking", lambda: None, lambda: ((1, 2), None))
        finally:
            unsubscribe()
        assert slept == [0.15]

    def test_a_failing_target_lookup_still_runs_the_action(self):
        ran: list[bool] = []
        unsubscribe = activity.subscribe(lambda event: None)

        def broken() -> computer.Target:
            raise RuntimeError("no desktop")

        try:
            computer._beat("Clicking", lambda: ran.append(True), broken)
        finally:
            unsubscribe()
        assert ran == [True]

    def test_a_point_target_is_used_as_given(self):
        assert computer._at(3, 4) == ((3, 4), None)

    def test_key_chords_read_like_a_person_writes_them(self):
        assert computer._pretty_keys("ctrl+s") == "Ctrl+S"
        assert computer._pretty_keys("enter") == "Enter"
