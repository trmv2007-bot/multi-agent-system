"""Browser test: a question typed into the dashboard comes back as an answer.

This drives the page a person actually uses, over the real HTTP and SSE routes
and the real graphs; only the model is scripted (``DASHBOARD_FAKE_MODEL``), so
the test needs no gateway, no key and no network.

Run it with ``python -m pytest e2e -q`` after ``python -m playwright install
chromium``. It is deliberately outside ``tests/`` so the unit suite stays
credential-free and browser-free.
"""

import os
import re
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
import uvicorn
from playwright.sync_api import Browser, Locator, Page, expect, sync_playwright

from src.config import settings
from src.dashboard import server

# The scripted answer is markdown, so the fragment is a run of plain prose in it
# rather than its first characters.
ANSWER_FRAGMENT = "the words are not researched"
COST_LINE = re.compile(r"^\d+ calls? · [\d.]+k? tokens · \d+s")
STARTUP_SECONDS = 30
TURN_SECONDS = 60_000
BUILT_LINE = re.compile(r"^built (\d+) sections? in (\d+(?:\.\d)?)s$")
ELAPSED_LINE = re.compile(r"^\d+(?:\.\d)?s$")
# What the planner's skeleton looks like: it must never be what a finished tab holds.
PLACEHOLDER_MARKERS = ("Writing…", 'class="eyebrow"', "opacity:.45")
SECTION_PHASE = re.compile(r"^(?:writing|fixing) section (\d+) of (\d+)$")
CARRIED_SECTIONS = "The page currently has these sections"
# The scripted planner always writes the same three sections.
SCRIPTED_SECTIONS = 3
TWO_TABS = 2

# The build's status line and checklist change many times in a few seconds, so
# the page is watched rather than polled: every text and every checklist state
# is kept, and no intermediate step goes unseen.
WATCH = """
window.__build = { statuses: [], done: [], running: [] };

function sample() {
  const line = document.getElementById("preview-state");
  const text = line ? line.textContent.trim() : "";
  if (text && window.__build.statuses[window.__build.statuses.length - 1] !== text) {
    window.__build.statuses.push(text);
  }
  const rows = Array.from(document.querySelectorAll("#build-checklist li"));
  const done = rows
    .filter((row) => row.dataset.state === "done")
    .map((row) => Number(row.dataset.index));
  for (const index of done) {
    if (!window.__build.done.includes(index)) window.__build.done.push(index);
  }
  window.__build.running.push(done.length);
}

new MutationObserver(sample).observe(document, {
  childList: true,
  subtree: true,
  characterData: true,
  attributes: true,
  attributeFilter: ["data-state"],
});
sample();
"""


@pytest.fixture(scope="module")
def dashboard_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Serve the dashboard on a free port with its model scripted."""
    state = tmp_path_factory.mktemp("dashboard")
    # The page's own state files belong in the temp dir, never in the repo.
    server.CHAT_FILE = state / "chats.json"
    server.STATE_FILE = state / "thread-state.json"
    server.NOTES_FILE = state / "dashboard-notes.json"
    server.BUILD_FILE = state / "dashboard-builds.json"
    server.CHATS.clear()
    server.THREAD_STATE.clear()
    server.RUNS.clear()
    _forget_builds()

    checkpoint_db = settings.checkpoint_db
    settings.checkpoint_db = None
    os.environ["DASHBOARD_FAKE_MODEL"] = "1"

    running = uvicorn.Server(
        uvicorn.Config(server.create_app(), host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=running.run, name="dashboard-e2e", daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + STARTUP_SECONDS
        while not running.started:
            if time.monotonic() > deadline:
                raise RuntimeError("the dashboard did not start")
            time.sleep(0.05)
        port = running.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        running.should_exit = True
        thread.join(timeout=STARTUP_SECONDS)
        os.environ.pop("DASHBOARD_FAKE_MODEL", None)
        settings.checkpoint_db = checkpoint_db


def _forget_builds() -> None:
    """Give each test a dashboard with no pages already open."""
    server.BUILDS.clear()
    server.BUILD_ORDER.clear()


@pytest.fixture(autouse=True)
def clean_dashboard() -> Iterator[None]:
    """Each test starts from an empty dashboard, not the last test's tabs."""
    _forget_builds()
    server.CHATS.clear()
    server.THREAD_STATE.clear()
    server.RUNS.clear()
    yield


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    """One Chromium for the whole module."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture()
def session(browser: Browser, dashboard_url: str) -> Iterator[tuple[Page, list[str]]]:
    """A fresh page with its console watched, plus every problem it reported."""
    context = browser.new_context()
    page = context.new_page()
    problems: list[str] = []
    page.on("console", lambda message: _note(problems, message))
    page.on("pageerror", lambda error: problems.append(f"page error: {error}"))
    page.add_init_script(WATCH)
    page.goto(dashboard_url)
    yield page, problems
    context.close()


def _note(problems: list[str], message: Any) -> None:
    """Record a console error, ignoring the browser's favicon request."""
    if getattr(message, "type", "") != "error":
        return
    text = str(getattr(message, "text", message))
    if "favicon" in text:
        return
    problems.append(text)


def ask(page: Page, question: str) -> None:
    """Type a question, send it, and wait for the turn to finish."""
    page.locator("#input").fill(question)
    page.locator("#send").click()
    expect(page.locator("#status-text")).to_have_text("done", timeout=TURN_SECONDS)


def last_reply(page: Page) -> Locator:
    """The answer bubble of the turn that just ran."""
    return page.locator(".turn.bot").last


def phase_of(status: str) -> tuple[int, int]:
    """The section number and the section total a numbered phase names."""
    match = SECTION_PHASE.match(status)
    assert match is not None, status
    return int(match.group(1)), int(match.group(2))


def expect_answer_and_cost(page: Page, question: str) -> None:
    """The question is on screen, the answer rendered, and the cost line is under it."""
    expect(page.locator(".turn.user").last).to_contain_text(question)
    reply = last_reply(page)
    expect(reply.locator(".bubble")).to_contain_text(ANSWER_FRAGMENT)
    expect(reply.locator(".cost")).to_have_text(COST_LINE)


def test_agent_mode_answers_with_a_cost_line(
    session: tuple[Page, list[str]],
) -> None:
    """The default mode: ask something, read the answer, read what it cost."""
    page, problems = session
    ask(page, "What does this dashboard do?")
    expect_answer_and_cost(page, "What does this dashboard do?")
    assert problems == []


def test_answer_renders_markdown(session: tuple[Page, list[str]]) -> None:
    """Bold, a table and a fenced block arrive as elements, not as characters."""
    page, problems = session
    ask(page, "What does the scripted answer look like?")
    body = last_reply(page).locator(".bubble .md")

    expect(body.locator("strong").first).to_have_text("Scripted demo answer.")
    table = body.locator("table")
    expect(table).to_have_count(1)
    expect(table.locator("thead th")).to_have_count(3)
    expect(table.locator("tbody tr")).to_have_count(3)
    expect(table.locator("tbody tr").first.locator("td")).to_have_count(3)
    expect(body.locator("pre code")).to_have_attribute("data-lang", "bash")
    expect(body.locator("pre code")).to_contain_text("OPENAI_API_KEY")

    # Nothing the model wrote as syntax is left on screen as syntax.
    visible = body.inner_text()
    assert "**" not in visible
    assert "| --- |" not in visible
    assert "```" not in visible
    # The cost line is the turn's metadata, so it stays outside the answer.
    expect(last_reply(page).locator(".cost")).to_have_text(COST_LINE)
    assert last_reply(page).locator(".md .cost").count() == 0
    assert problems == []


def test_research_tab_runs_a_turn(session: tuple[Page, list[str]]) -> None:
    """The Research tab switches the mode, the prompt and the run behind it."""
    page, problems = session

    page.locator("#mode-research").click()
    expect(page.locator("#mode-research")).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#mode-agent")).to_have_attribute("aria-pressed", "false")
    expect(page.locator("#input")).not_to_have_attribute(
        "placeholder", "Tell the agent what to do…"
    )

    ask(page, "What is worth researching today?")
    expect_answer_and_cost(page, "What is worth researching today?")
    assert problems == []


def test_build_tab_builds_a_page(session: tuple[Page, list[str]]) -> None:
    """The Build tab shows the preview stage and fills it with a page."""
    page, problems = session

    page.locator("#mode-build").click()
    expect(page.locator("#mode-build")).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#preview-stage")).to_have_class(re.compile(r"\bon\b"))
    expect(page.locator("#input")).to_have_attribute(
        "placeholder", "Describe a new page — it opens its own tab…"
    )

    ask(page, "A page about the dashboard")
    expect(page.locator("#preview-state")).to_have_attribute("data-state", "built")
    expect(page.locator("#preview-code")).to_contain_text("<section")
    expect(last_reply(page).locator(".cost")).to_have_text(COST_LINE)
    assert problems == []


def test_build_reports_progress_a_person_can_read(
    session: tuple[Page, list[str]],
) -> None:
    """The status line and checklist say what the build is doing, then that it stopped."""
    page, problems = session

    page.locator("#mode-build").click()
    ask(page, "A page about the dashboard")

    watched = page.evaluate("window.__build")
    statuses: list[str] = watched["statuses"]

    # It starts by saying what it is doing, before any section exists.
    assert "planning" in statuses, statuses
    numbered = [status for status in statuses if SECTION_PHASE.match(status)]
    assert numbered, statuses
    assert statuses.index("planning") < statuses.index(numbered[0])

    total = phase_of(numbered[0])[1]
    assert total >= SCRIPTED_SECTIONS
    seen = [phase_of(status)[0] for status in numbered]
    assert seen == sorted(seen), statuses
    # The plan's last section is the last one named; the scripted build is fast
    # enough that a started section can be replaced before the page paints it.
    assert 1 <= seen[0] <= total and max(seen) == total, statuses

    # Every planned section was listed before it landed, and finished in order.
    expect(page.locator("#build-checklist li")).to_have_count(total)
    states = page.evaluate(
        "[...document.querySelectorAll('#build-checklist li')].map((row) => row.dataset.state)"
    )
    assert states == ["done"] * total
    assert watched["done"] == list(range(total))
    assert watched["running"] == sorted(watched["running"])

    # It ends with an unmistakable line: how many sections, and how long.
    built = BUILT_LINE.match(statuses[-1])
    assert built, statuses[-1]
    assert int(built.group(1)) == total
    expect(page.locator("#preview-state")).to_have_attribute("data-state", "built")
    expect(page.locator("#build-elapsed")).to_have_text(ELAPSED_LINE)
    assert problems == []


def test_a_second_build_opens_its_own_tab(
    session: tuple[Page, list[str]], dashboard_url: str
) -> None:
    """A new build is a new tab with its own page, never an edit of the old one."""
    page, problems = session

    page.locator("#mode-build").click()
    ask(page, "A page about the dashboard")

    tabs = page.locator("#build-tabs .build-tab")
    expect(tabs).to_have_count(1)
    first_document = page.locator("#preview-code").text_content()
    assert first_document and "<section" in first_document

    ask(page, "A page about the second dashboard")

    expect(tabs).to_have_count(TWO_TABS)
    expect(tabs.nth(1)).to_have_attribute("aria-selected", "true")
    expect(tabs.nth(0)).to_have_attribute("aria-selected", "false")
    second_document = page.locator("#preview-code").text_content()
    assert second_document and second_document != first_document
    for document in (first_document, second_document):
        for marker in PLACEHOLDER_MARKERS:
            assert marker not in document, marker

    # Switching back is instant and shows exactly what the first build left.
    tabs.nth(0).click()
    expect(tabs.nth(0)).to_have_attribute("aria-selected", "true")
    expect(tabs.nth(1)).to_have_attribute("aria-selected", "false")
    assert page.locator("#preview-code").text_content() == first_document

    # Neither request carried the other page's sections into the planner.
    builds = page.request.get(f"{dashboard_url}/api/builds").json()["builds"]
    assert len(builds) == TWO_TABS
    assert [row["document"] for row in builds] == [first_document, second_document]
    for row in builds:
        assert CARRIED_SECTIONS not in row["request"], row["request"]
    assert builds[0]["thread_id"] != builds[1]["thread_id"]
    assert problems == []


def test_editing_a_page_carries_its_sections(
    session: tuple[Page, list[str]], dashboard_url: str
) -> None:
    """The explicit edit is the one path that keeps the page it is changing."""
    page, problems = session

    page.locator("#mode-build").click()
    ask(page, "A page about the dashboard")
    page.locator("#build-edit").click()
    expect(page.locator("#build-edit")).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#input")).to_have_attribute(
        "placeholder", "Ask for a change to this page…"
    )

    ask(page, "Make the headline louder")

    expect(page.locator("#build-tabs .build-tab")).to_have_count(1)
    expect(page.locator("#preview-state")).to_have_attribute("data-state", "built")
    builds = page.request.get(f"{dashboard_url}/api/builds").json()["builds"]
    assert len(builds) == 1
    assert CARRIED_SECTIONS in builds[0]["request"]
    assert problems == []


def test_tabs_survive_a_refresh(session: tuple[Page, list[str]], dashboard_url: str) -> None:
    """A refresh brings back every tab and the page each one holds."""
    page, problems = session

    page.locator("#mode-build").click()
    ask(page, "A page about the dashboard")
    ask(page, "A page about the second dashboard")

    titles = page.locator("#build-tabs .build-tab .label").all_text_contents()
    documents = page.locator("#preview-code").text_content()
    assert len(titles) == TWO_TABS

    page.reload()

    expect(page.locator("#mode-build")).to_have_attribute("aria-pressed", "true")
    tabs = page.locator("#build-tabs .build-tab")
    expect(tabs).to_have_count(TWO_TABS)
    assert page.locator("#build-tabs .build-tab .label").all_text_contents() == titles
    expect(tabs.nth(1)).to_have_attribute("aria-selected", "true")
    assert page.locator("#preview-code").text_content() == documents

    tabs.nth(0).click()
    expect(tabs.nth(0)).to_have_attribute("aria-selected", "true")
    assert page.locator("#preview-code").text_content() != documents
    assert problems == []
