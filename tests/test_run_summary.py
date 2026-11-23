"""Per-run accounting: what one turn cost, with no network and no credential.

The numbers a run reports must match the calls it actually made, so each test
drives a real graph or a real JSON call with a scripted model and then reads
the summary the CLI and the dashboard both print.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from evals.harness import Meter
from evals.stub import GROUNDED, QUESTION, ScriptedModel
from src.agents.verify import SUPPORTED
from src.graph.workflow import run_workflow
from src.observability import end_run, format_run_summary, run_summary, start_run
from src.structured import STRUCTURED, call_json

PROMPT_TOKENS = 120
COMPLETION_TOKENS = 30

SUMMARY_KEYS = {
    "label",
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "tokens",
    "seconds",
    "model_seconds",
    "slowest_agent",
    "agents",
    "models",
    "structured",
}


@pytest.fixture(autouse=True)
def fresh_books():
    """No test inherits the run another test was accounting."""
    end_run()
    yield
    end_run()


class CountingScripted(ScriptedModel):
    """The eval script, reporting the usage numbers a real provider reports."""

    async def ainvoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Answer from the script, with usage attached to the reply."""
        response = await super().ainvoke(messages, **kwargs)
        response.usage_metadata = {
            "input_tokens": PROMPT_TOKENS,
            "output_tokens": COMPLETION_TOKENS,
        }
        return response


async def test_counters_accumulate_across_a_multi_agent_run() -> None:
    """One pipeline run books every agent's calls, tokens, and the provider that served them."""
    meter = Meter()
    model = CountingScripted(SUPPORTED, GROUNDED, meter)
    start_run("pipeline test")

    await run_workflow(QUESTION, "cost-test", llm=model, checkpointer=MemorySaver(), fresh=True)
    summary = run_summary()

    assert summary["calls"] == meter.calls >= 3
    assert summary["prompt_tokens"] == meter.calls * PROMPT_TOKENS
    assert summary["completion_tokens"] == meter.calls * COMPLETION_TOKENS
    assert summary["tokens"] == meter.calls * (PROMPT_TOKENS + COMPLETION_TOKENS)
    assert summary["model_seconds"] <= summary["seconds"]
    assert len(summary["models"]) == 1
    assert summary["models"][0]["calls"] == meter.calls
    assert summary["models"][0]["tokens"] == summary["tokens"]


async def test_every_agent_gets_its_own_line() -> None:
    """Agents are accounted separately, and the slowest one is named."""
    meter = Meter()
    model = CountingScripted(SUPPORTED, GROUNDED, meter)
    start_run("pipeline test")

    await run_workflow(QUESTION, "cost-agents", llm=model, checkpointer=MemorySaver(), fresh=True)
    summary = run_summary()

    assert len(summary["agents"]) >= 3
    assert sum(agent["calls"] for agent in summary["agents"]) == summary["calls"]
    assert sum(agent["tokens"] for agent in summary["agents"]) == summary["tokens"]
    assert summary["slowest_agent"] == summary["agents"][0]["name"]
    assert summary["structured"]["calls"] > 0


def test_an_empty_run_reports_zeros() -> None:
    """A run that made no calls has a full summary of zeros rather than an error."""
    start_run("nothing happened")
    summary = run_summary()

    assert summary["calls"] == 0
    assert summary["prompt_tokens"] == 0
    assert summary["completion_tokens"] == 0
    assert summary["tokens"] == 0
    assert summary["seconds"] >= 0
    assert summary["slowest_agent"] == ""
    assert summary["agents"] == []
    assert summary["models"] == []
    assert summary["structured"]["repaired"] == 0


def test_no_run_at_all_reports_zeros() -> None:
    """Reading the summary outside a run is not an error."""
    assert run_summary()["calls"] == 0


def test_summary_shape_is_stable() -> None:
    """The keys the dashboard and the CLI read do not drift."""
    start_run("shape")
    summary = run_summary()

    assert set(summary) == SUMMARY_KEYS
    assert set(summary["structured"]) == set(STRUCTURED.FIELDS)
    assert format_run_summary(summary).startswith("0 model calls")


async def test_a_repaired_json_reply_is_counted() -> None:
    """A reply that only parses after repair shows up in the run's books."""

    async def unquoted_value(_messages: list[dict[str, Any]]) -> Any:
        return SimpleNamespace(content='{"verdict": supported}')

    start_run("repair test")
    payload = await call_json(
        unquoted_value, [{"role": "user", "content": "verdict?"}], site="test"
    )
    summary = run_summary()

    assert payload == {"verdict": "supported"}
    assert summary["structured"]["calls"] == 1
    assert summary["structured"]["repaired"] == 1
    assert "1 JSON repair" in format_run_summary(summary)
