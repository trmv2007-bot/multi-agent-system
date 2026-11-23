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


@pytest.fixture(scope="module")
def dashboard_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Serve the dashboard on a free port with its model scripted."""
    state = tmp_path_factory.mktemp("dashboard")
    # The page's own state files belong in the temp dir, never in the repo.
    server.CHAT_FILE = state / "chats.json"
    server.STATE_FILE = state / "thread-state.json"
    server.NOTES_FILE = state / "dashboard-notes.json"
    server.CHATS.clear()
    server.THREAD_STATE.clear()
    server.RUNS.clear()

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
    expect(page.locator("#input")).to_have_attribute("placeholder", "Ask for a change to the page…")

    ask(page, "A page about the dashboard")
    expect(page.locator("#preview-state")).to_have_text("built")
    expect(page.locator("#preview-code")).to_contain_text("<section")
    expect(last_reply(page).locator(".cost")).to_have_text(COST_LINE)
    assert problems == []
