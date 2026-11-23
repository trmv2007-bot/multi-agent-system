"""What a build's status line and checklist say while it runs.

The page draws these states, and the browser test drives the real thing; these
cover the transitions themselves, including the ones a scripted build is too
fast to reach in a browser.
"""

from src.dashboard.build_progress import DONE, PENDING, WRITING, BuildProgress, clock


def test_a_build_starts_by_saying_it_is_planning() -> None:
    """Before the planner answers there is nothing else to say."""
    progress = BuildProgress()
    assert progress.status == "planning"
    assert progress.checklist() == []
    assert progress.running is True


def test_the_plan_is_listed_before_any_section_is_written() -> None:
    """The shape of the work is visible while all of it is still pending."""
    progress = BuildProgress()
    progress.saw_plan(["hero", "details", "close"])

    assert progress.status == "planned 3 sections"
    assert [item["state"] for item in progress.checklist()] == [PENDING] * 3
    assert [item["name"] for item in progress.checklist()] == ["hero", "details", "close"]


def test_sections_are_named_while_they_are_written_and_after() -> None:
    """Each phase names the section it is on, counting through the plan."""
    progress = BuildProgress()
    progress.saw_plan(["hero", "details", "close"])

    progress.saw_worker(0)
    assert progress.status == "writing section 1 of 3"
    assert progress.checklist()[0]["state"] == WRITING

    progress.saw_section(0)
    assert progress.status == "writing section 2 of 3"
    assert progress.sections == 1

    progress.saw_section(1)
    assert progress.status == "writing section 3 of 3"

    progress.saw_section(2)
    assert progress.status == "reviewing"
    assert progress.sections == 3


def test_a_repair_is_named_as_a_repair() -> None:
    """After the review, a section landing again is a fix, not a draft."""
    progress = BuildProgress()
    progress.saw_plan(["hero", "details"])
    progress.saw_section(0)
    progress.saw_section(1)
    progress.saw_review()

    progress.saw_worker(0)
    assert progress.status == "fixing section 1 of 2"

    progress.saw_section(0)
    assert progress.status == "fixed section 1 of 2"
    assert progress.checklist()[0]["state"] == DONE


def test_a_finished_build_says_how_much_and_how_long() -> None:
    """The last line is a count and a time, so nothing looks still running."""
    progress = BuildProgress()
    progress.saw_plan(["hero", "details", "close"])
    for index in range(3):
        progress.saw_section(index)
    progress.built(42.4)

    assert progress.status == "built 3 sections in 42s"
    assert progress.running is False
    assert progress.phase == "built"


def test_a_short_build_is_never_reported_as_zero() -> None:
    """A real build takes a while, but a fast one must not read as "in 0s"."""
    progress = BuildProgress()
    progress.saw_plan(["hero"])
    progress.saw_section(0)
    progress.built(0.34)

    assert progress.status == "built 1 section in 0.3s"


def test_a_long_build_reads_in_whole_seconds() -> None:
    """Once seconds matter, the decimal is noise."""
    assert clock(42.4) == "42s"
    assert clock(9.96) == "10.0s"
    assert clock(10.0) == "10s"


def test_a_stopped_build_says_how_far_it_got() -> None:
    """A failure is unmistakable, and it says how many sections are missing."""
    progress = BuildProgress()
    progress.saw_plan(["hero", "details", "close"])
    progress.saw_section(0)
    progress.saw_section(1)
    progress.stopped(8.0)

    assert progress.status == "stopped after 2 of 3 sections"
    assert progress.running is False
    assert progress.phase == "failed"


def test_progress_survives_being_saved() -> None:
    """A refresh rebuilds the same checklist from what was stored."""
    progress = BuildProgress(started=10.0)
    progress.saw_plan(["hero", "details", "close"])
    progress.saw_section(0)

    restored = BuildProgress.from_dict({**progress.as_dict(), "items": progress.items})

    assert restored.plan == ["hero", "details", "close"]
    assert restored.checklist() == progress.checklist()
    assert restored.status == progress.status
    assert restored.sections == 1
