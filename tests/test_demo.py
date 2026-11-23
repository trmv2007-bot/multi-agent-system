"""The scripted demo provider: every mode runs, and none of it needs a key."""

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.agents.verify import SUPPORTED, parse_verdicts
from src.config import get_llm, settings
from src.demo import DemoModel, demo_model
from src.graph.agent import run_agent
from src.graph.build import run_build
from src.graph.research import run_research
from src.graph.team import run_team
from src.graph.workflow import run_workflow

QUERY = "Summarise what this system does"

MODES = {
    "agent": lambda model: run_agent(QUERY, "demo-agent", model, checkpointer=MemorySaver()),
    "pipeline": lambda model: run_workflow(
        QUERY, "demo-pipeline", model, checkpointer=MemorySaver()
    ),
    "team": lambda model: run_team(
        QUERY, "demo-team", model, max_subtasks=2, checkpointer=MemorySaver()
    ),
    "research": lambda model: run_research(
        QUERY, "demo-research", model, max_questions=3, waves=1, checkpointer=MemorySaver()
    ),
    "build": lambda model: run_build(
        QUERY, "demo-build", model, max_sections=3, checkpointer=MemorySaver()
    ),
}


@pytest.mark.parametrize("mode", sorted(MODES))
async def test_every_mode_answers_without_a_key(mode: str) -> None:
    """The scripted model drives each real graph to an answer, with no errors."""
    state = await MODES[mode](DemoModel())

    assert state.final_report.strip()
    assert state.errors == []


async def test_pipeline_runs_the_real_validator() -> None:
    """The demo answer was graded against the notes, so the grade is not a stub."""
    state = await run_workflow(QUERY, "demo-verify", DemoModel(), checkpointer=MemorySaver())

    verdicts = {item.get("verdict") for item in state.validation_results}

    assert SUPPORTED in verdicts
    assert state.context.get("verification") != "abstained"


def test_verdicts_are_one_per_claim() -> None:
    """A validator prompt with three claims gets three readable verdicts."""
    prompt = [{"role": "user", "content": "Check every claim below.\nCLAIM 0\nCLAIM 1\nCLAIM 2\n"}]

    verdicts = parse_verdicts(DemoModel().reply(prompt), count=3)

    assert [item["verdict"] for item in verdicts if item] == [SUPPORTED] * 3


def test_team_plan_is_json_the_planner_can_read() -> None:
    """The orchestrator's scripted reply is the JSON shape agents/team.py parses."""
    prompt = [{"role": "system", "content": "You are the orchestrator of a team of agents"}]

    plan: Any = json.loads(DemoModel().reply(prompt))

    assert [item["task"] for item in plan]


def test_demo_mode_routes_get_llm_to_the_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """With demo_mode on, every caller gets the scripted provider."""
    monkeypatch.setattr(settings, "demo_mode", True)

    assert get_llm() is demo_model()


def test_main_serves_the_dashboard_in_demo_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The demo entry point turns demo mode on and hands off to the dashboard."""
    from src import demo

    served: list[str] = []
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr("src.dashboard.server.main", lambda: served.append("dashboard"))

    demo.main()

    assert served == ["dashboard"]
    assert settings.demo_mode is True
