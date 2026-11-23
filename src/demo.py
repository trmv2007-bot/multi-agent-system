"""A scripted provider, so the dashboard runs with no model key at all.

``multi-agent-demo`` serves the same dashboard as ``multi-agent-dashboard``, but
every model call is answered from the script below instead of a gateway. The
graph, the validator, the checkpointer, the activity feed and the cost line all
run for real; only the model is replaced. Nothing here opens a socket or spends
anything, and nothing it says is a fact about the world: the point is to see the
interface working before any credential exists.

:func:`demo_model` is what ``get_llm()`` returns while ``demo_mode`` is set, so
the CLI, the dashboard and the MCP server all reach it the same way.
"""

import json
import re
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

DEMO_PROVIDER = "demo"
DEMO_MODEL_ID = "scripted"

DEMO_BANNER = (
    "Demo mode: no model key needed.\n"
    "  Every model call is answered from src/demo.py, so the answers are scripted\n"
    "  and nothing leaves this machine. Ask anything: the graph, the validator,\n"
    "  the cost line and the dashboard are all real.\n"
)

DEMO_ANSWER = (
    "Scripted demo answer. No model key is configured, so every call in this run "
    "was answered from src/demo.py: the graph, the validator and the cost meter "
    "are real, the words are not researched. Add a key to .env and the same run "
    "goes to a live model."
)

RESEARCH_NOTES = (
    "Scripted research notes for the demo provider. A live run would search the "
    "web, open the pages it found and keep the quotable facts; this run keeps a "
    "placeholder so the validator, the writer and the cost line all have "
    "something to work on."
)

ANALYSIS = (
    "The scripted notes carry one finding and no numbers, so there is nothing to "
    "contradict. A live analysis would separate what the sources established from "
    "what they only implied."
)

FINDING = "A scripted finding: the demo provider answers without touching the network."

SOURCES = "SOURCES:\n- https://example.com/demo-provider"

TEAM_WORKER = (
    "Scripted subtask result. A live worker would use its tools for this subtask "
    "and report what it actually found."
)

TEAM_SYNTHESIS = (
    "Scripted team answer. The planner split the task, one worker ran per "
    "subtask, and this line is what the synthesizer was given. The graph is real; "
    "the words come from src/demo.py."
)

SECTION_HTML = (
    '<section class="container">\n'
    "  <{heading}>{section}</{heading}>\n"
    '  <p class="sub">{brief}</p>\n'
    "</section>"
)

SECTION_RE = re.compile(r"Your section: ([^\n]*)")
BRIEF_RE = re.compile(r"Brief: ([^\n]*)")

TEAM_PLAN = json.dumps(
    [
        {"task": "Establish what the task actually needs", "tools": []},
        {"task": "Draft the part that can be written from what is already known", "tools": []},
    ]
)

RESEARCH_PLAN = json.dumps(
    [
        {"question": "What does the task ask for?", "why": "the task itself"},
        {"question": "What is already known about it?", "why": "the starting point"},
        {"question": "What is missing?", "why": "what the answer must add"},
    ]
)

PAGE_PLAN = json.dumps(
    [
        {"section": "hero", "brief": "the headline, one line of subhead and the call to action"},
        {"section": "details", "brief": "what the thing is, in a short paragraph"},
        {"section": "close", "brief": "the final call to action"},
    ]
)

PAGE_THEME = json.dumps(
    {
        "feeling": "calm and direct",
        "radius": "10px",
        "display_size": "clamp(2rem, 5vw, 3.2rem)",
    }
)

SCRIPT: tuple[tuple[str, str], ...] = (
    ("You review an assembled web page", "[]"),
    ("One section of a page needs a specific fix.", SECTION_HTML),
    ("You are the designer on this build", PAGE_THEME),
    ("You plan one web page.", PAGE_PLAN),
    ("You research what a web page needs before it is written", RESEARCH_NOTES + "\n\n" + SOURCES),
    ("You report on a page the team just built", DEMO_ANSWER),
    ("You are the orchestrator of a team", TEAM_PLAN),
    ("You are one specialist agent on a team", TEAM_WORKER),
    ("You merge the work of a team", TEAM_SYNTHESIS),
    ("You plan research.", RESEARCH_PLAN),
    ("You are one researcher on a team", RESEARCH_NOTES + "\n\n" + SOURCES),
    ("You review research coverage", "[]"),
    ("You write the final answer from a team's research", DEMO_ANSWER),
    ("You are a Research Agent", RESEARCH_NOTES + "\n\n" + SOURCES),
    ("You are an Analysis Agent", ANALYSIS),
    ("You are the writing agent", DEMO_ANSWER),
)

VERDICT_MARKERS = ("You verify claims against source text", "Check every claim below")


def _prompt_of(messages: Any) -> str:
    """Every message flattened into one string, which is what the script keys on."""
    return "\n".join(str(step.get("content", "")) for step in messages if hasattr(step, "get"))


def _option(prompt: str, pattern: re.Pattern[str], fallback: str) -> str:
    """One captured value from the prompt, or the fallback."""
    match = pattern.search(prompt)
    return match.group(1).strip() if match and match.group(1).strip() else fallback


def _section_fragment(prompt: str) -> str:
    """A finished-looking section using only the classes the stylesheet defines."""
    section = _option(prompt, SECTION_RE, "section")
    heading = "h1" if "hero" in section.lower() else "h2"
    return SECTION_HTML.format(
        heading=heading, section=section, brief=_option(prompt, BRIEF_RE, "a short brief")
    )


def _verdicts(prompt: str) -> str:
    """One supported verdict per claim the validator listed."""
    claims = prompt.count("CLAIM ")
    return json.dumps(
        [
            {"index": index, "verdict": "supported", "reason": "scripted demo provider"}
            for index in range(max(claims, 1))
        ]
    )


class DemoModel:
    """Answers each agent's prompt from a script and never touches the network."""

    def bind(self, **_options: Any) -> "DemoModel":
        """Accept request options; the script does not change with them."""
        return self

    def bind_tools(self, _tools: Any) -> "DemoModel":
        """Accept tools; the script never calls one."""
        return self

    async def ainvoke(self, messages: list[dict[str, Any]], **_options: Any) -> Any:
        """Answer one prompt by role."""
        text = self.reply(messages)
        return SimpleNamespace(
            content=text,
            tool_calls=None,
            response_metadata={"model_name": f"{DEMO_PROVIDER}/{DEMO_MODEL_ID}"},
            usage_metadata={
                "input_tokens": len(_prompt_of(messages)) // 4,
                "output_tokens": len(text) // 4,
                "total_tokens": (len(_prompt_of(messages)) + len(text)) // 4,
            },
        )

    def reply(self, messages: list[dict[str, Any]]) -> str:
        """The scripted text for one prompt."""
        prompt = _prompt_of(messages)
        if any(marker in prompt for marker in VERDICT_MARKERS):
            return _verdicts(prompt)
        if "Write one section of a web page as an HTML fragment" in prompt:
            return _section_fragment(prompt)
        for marker, answer in SCRIPT:
            if marker in prompt:
                return answer
        # One agent working a task end to end, or anything new: still an answer.
        return DEMO_ANSWER


@lru_cache(maxsize=1)
def demo_model() -> DemoModel:
    """The one scripted provider this process uses."""
    return DemoModel()


def main() -> None:
    """Serve the dashboard with every model call scripted."""
    from src.config import settings
    from src.dashboard.server import main as serve_dashboard

    settings.demo_mode = True
    print(DEMO_BANNER)
    serve_dashboard()


if __name__ == "__main__":
    main()
