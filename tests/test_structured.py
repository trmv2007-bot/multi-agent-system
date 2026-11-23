"""One parser for every structured reply, proved against each way JSON breaks.

The provider pool returns JSON with unquoted values, and returns empty content
when a reasoning model spends the budget on thinking. Every test here stubs the
model, so none of them touch the network.
"""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.activity import activity
from src.observability import structured_output_stats
from src.structured import (
    JSON_EXTRA_BODY,
    JSON_MAX_TOKENS,
    STRUCTURED,
    StructuredOutputError,
    call_json,
    parse_json,
    read_json,
)

PLAN = '[{"task": "one", "tools": []}, {"task": "two", "tools": []}]'


class StubModel:
    """A model that answers from a queue of replies and records every prompt."""

    def __init__(self, *replies: str | None) -> None:
        self.replies = list(replies)
        self.prompts: list[list[dict[str, Any]]] = []
        self.binds: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any) -> "StubModel":
        self.binds.append(kwargs)
        return self

    async def ainvoke(self, messages: list[dict[str, Any]], **_kwargs: Any) -> Any:
        self.prompts.append(messages)
        return SimpleNamespace(content=self.replies.pop(0) if self.replies else "", tool_calls=None)


@pytest.fixture(autouse=True)
def fresh_counters():
    """Counters are process-wide, so no test may inherit another's tallies."""
    STRUCTURED.reset()
    yield
    STRUCTURED.reset()


def asked() -> list[dict[str, Any]]:
    """The messages a caller would send when it wants JSON back."""
    return [{"role": "user", "content": "Break this into subtasks as JSON."}]


class TestCallJsonSuccess:
    """A reply that is already JSON costs one call and no repair."""

    @pytest.mark.asyncio
    async def test_clean_reply_is_read_once_and_without_repair(self):
        model = StubModel(PLAN)

        plan = await call_json(model, asked(), site="test.planner")

        assert plan == json.loads(PLAN)
        assert len(model.prompts) == 1
        assert STRUCTURED.snapshot()["parsed"] == 1
        assert STRUCTURED.snapshot()["repaired"] == 0

    @pytest.mark.asyncio
    async def test_the_call_asks_for_no_reasoning_and_a_generous_budget(self):
        model = StubModel(PLAN)

        await call_json(model, asked(), site="test.planner")

        assert model.binds == [{"max_tokens": JSON_MAX_TOKENS, "extra_body": JSON_EXTRA_BODY}]
        assert JSON_EXTRA_BODY["reasoning_effort"] == "none"

    @pytest.mark.asyncio
    async def test_a_sender_owns_its_own_tuning(self):
        calls: list[list[dict[str, Any]]] = []

        async def send(messages: list[dict[str, Any]]) -> Any:
            calls.append(messages)
            return SimpleNamespace(content=PLAN)

        assert await call_json(send, asked(), site="test.planner") == json.loads(PLAN)
        assert len(calls) == 1


class TestCallJsonRepair:
    """Each way a gateway breaks JSON is repaired, and the repair is counted."""

    @pytest.mark.asyncio
    async def test_a_fenced_reply_is_read(self):
        model = StubModel(f"Here is the plan:\n```json\n{PLAN}\n```\nAnything else?")

        assert await call_json(model, asked(), site="test.planner") == json.loads(PLAN)
        assert len(model.prompts) == 1

    @pytest.mark.asyncio
    async def test_prose_before_the_object_is_skipped(self):
        reply = f"Sure [the plan] is below.\n{PLAN}\nLet me know if you want changes."
        model = StubModel(reply)

        assert await call_json(model, asked(), site="test.planner") == json.loads(PLAN)
        assert len(model.prompts) == 1
        assert STRUCTURED.snapshot()["repaired"] == 0

    @pytest.mark.asyncio
    async def test_unquoted_values_are_repaired_and_counted(self):
        # Observed from a pooled model: the values lost their quotes, so the
        # reply is not JSON, but every field in it is unambiguous.
        reply = '[{"task": two, "tools": []}, {"task": "three", "note": Source says otherwise."}]'
        model = StubModel(reply)

        plan = await call_json(model, asked(), site="test.planner")

        assert plan == [
            {"task": "two", "tools": []},
            {"task": "three", "note": "Source says otherwise."},
        ]
        assert len(model.prompts) == 1
        assert STRUCTURED.snapshot()["repaired"] == 1
        assert STRUCTURED.snapshot()["parsed"] == 0

    @pytest.mark.asyncio
    async def test_trailing_comma_is_repaired_and_counted(self):
        model = StubModel('[{"task": "one",},]')

        assert await call_json(model, asked(), site="test.planner") == [{"task": "one"}]
        assert STRUCTURED.snapshot()["repaired"] == 1

    def test_a_bare_value_keeps_its_own_commas(self):
        payload, repaired = read_json('[{"reason": Source is 389,444 strong.}]')

        assert payload == [{"reason": "Source is 389,444 strong."}]
        assert repaired is True


class TestCallJsonFailure:
    """A reply with no JSON in it is a failure, never a result."""

    @pytest.mark.asyncio
    async def test_empty_content_is_a_failure_that_a_retry_can_fix(self):
        model = StubModel("", PLAN)

        assert await call_json(model, asked(), site="test.planner") == json.loads(PLAN)
        assert len(model.prompts) == 2
        assert "JSON only" in model.prompts[1][-1]["content"]
        counts = STRUCTURED.snapshot()
        assert counts["empty"] == 1
        assert counts["retried"] == 1

    @pytest.mark.asyncio
    async def test_empty_content_never_becomes_a_result(self):
        model = StubModel(None, "   \n")

        with pytest.raises(StructuredOutputError) as caught:
            await call_json(model, asked(), retries=1, site="test.planner")

        assert caught.value.attempts == 2
        assert caught.value.raw.strip() == ""
        assert "empty content" in str(caught.value)
        counts = STRUCTURED.snapshot()
        assert counts["empty"] == 2
        assert counts["failed"] == 1

    @pytest.mark.asyncio
    async def test_a_truncated_object_raises_and_carries_the_reply(self):
        truncated = '[{"task": "one", "tools": []}, {"task": "tw'
        model = StubModel(truncated, truncated)

        with pytest.raises(StructuredOutputError) as caught:
            await call_json(model, asked(), retries=1, site="test.planner")

        assert caught.value.raw == truncated
        assert caught.value.site == "test.planner"
        assert STRUCTURED.snapshot()["unreadable"] == 2

    @pytest.mark.asyncio
    async def test_a_reply_that_never_repairs_raises_instead_of_returning_junk(self):
        model = StubModel("I could not decide how to split this.", "Still no idea, sorry.")

        with pytest.raises(StructuredOutputError) as caught:
            await call_json(model, asked(), site="test.planner")

        assert "no readable JSON" in str(caught.value)
        assert STRUCTURED.snapshot()["failed"] == 1

    @pytest.mark.asyncio
    async def test_no_retry_means_one_call(self):
        model = StubModel("")

        with pytest.raises(StructuredOutputError):
            await call_json(model, asked(), retries=0, site="test.planner")

        assert len(model.prompts) == 1
        assert STRUCTURED.snapshot()["retried"] == 0

    @pytest.mark.asyncio
    async def test_the_expected_shape_filters_what_counts(self):
        model = StubModel(PLAN)

        with pytest.raises(StructuredOutputError):
            await call_json(model, asked(), expect=dict, site="test.planner")

    def test_parse_json_degrades_to_none_for_a_caller_with_a_fallback(self):
        assert parse_json("no json here") is None
        assert parse_json(PLAN) == json.loads(PLAN)


class TestVisibility:
    """A provider that has started mangling JSON must not be invisible."""

    @pytest.mark.asyncio
    async def test_repairs_are_published_on_the_activity_bus(self):
        events: list[dict[str, Any]] = []
        activity.subscribe(events.append)
        model = StubModel('[{"task": one}]')

        await call_json(model, asked(), site="test.planner")

        [event] = [item for item in events if item.get("type") == "structured"]
        assert event["site"] == "test.planner"
        assert event["counts"]["repaired"] == 1
        assert structured_output_stats() == STRUCTURED.snapshot()
