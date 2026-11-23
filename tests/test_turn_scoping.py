"""A follow-up turn must be answered, and validated, from its own work only.

Both graphs resume a thread's saved state with ``fresh=False`` in the dashboard,
so the records an earlier turn wrote are still in the state when the next one
starts. Every reader of those records has to ask for the current turn's.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver


def _router(llm):
    """A provider chain with one provider, so no test reaches the network."""
    stub = MagicMock()
    stub.provider_names.return_value = ["stub"]
    stub.next_provider.return_value = llm
    return stub


def _prompt(messages) -> str:
    """The text a fake model saw, whether the messages are dicts or langchain ones."""
    parts = []
    for message in messages:
        content = message.get("content", "") if isinstance(message, dict) else ""
        parts.append(str(getattr(message, "content", content)))
    return "\n".join(parts)


class TestTeamTurnScoping:
    """Team mode's synthesizer must merge this turn's workers, not the last one's."""

    @staticmethod
    def _llm(plans):
        queue = list(plans)
        prompts: list[str] = []

        async def ainvoke(messages, **kwargs):
            text = _prompt(messages)
            prompts.append(text)
            if "orchestrator of a team" in text:
                return MagicMock(content=queue.pop(0), tool_calls=None, usage_metadata={})
            if "merge the work of a team" in text:
                return MagicMock(content="synthesized", tool_calls=None, usage_metadata={})
            subtask = text.rsplit("Your subtask:", 1)[-1].strip()
            return MagicMock(content=f"work on {subtask}", tool_calls=None, usage_metadata={})

        llm = AsyncMock()
        llm.ainvoke = ainvoke
        llm.bind_tools = MagicMock(return_value=llm)
        return llm, prompts

    @pytest.mark.asyncio
    async def test_second_turn_synthesizes_only_its_own_workers(self):
        from src.graph.team import run_team

        llm, prompts = self._llm(
            [
                json.dumps([{"task": "postgres tradeoffs"}, {"task": "sqlite tradeoffs"}]),
                json.dumps([{"task": "coffee hero copy"}]),
            ]
        )
        saver = MemorySaver()

        with (
            patch("src.agents.base.get_llm", return_value=llm),
            patch("src.agents.base.router", _router(llm)),
        ):
            first = await run_team(
                "compare postgres and sqlite", "team-turns", llm, checkpointer=saver
            )
            second = await run_team(
                "write a coffee shop hero", "team-turns", llm, checkpointer=saver
            )

        assert len(first.subtask_results) == 2
        # The thread keeps the earlier turn's results; the answer must not.
        assert len(second.subtask_results) == 3

        merged = [p for p in prompts if "merge the work of a team" in p][-1]
        assert "coffee hero copy" in merged
        assert "postgres tradeoffs" not in merged
        assert "sqlite tradeoffs" not in merged
        assert merged.count("[Subtask 1]") == 1


class TestPipelineTurnScoping:
    """The research pipeline must stamp its records and read only this turn's."""

    @staticmethod
    def _llm():
        prompts: list[str] = []

        async def ainvoke(messages, **kwargs):
            text = _prompt(messages)
            prompts.append(text)
            answer = "notes on coffee" if "Question two" in text else "notes on postgres"
            return MagicMock(content=answer, tool_calls=None, usage_metadata={})

        llm = AsyncMock()
        llm.ainvoke = ainvoke
        llm.bind_tools = MagicMock(return_value=llm)
        return llm, prompts

    @pytest.mark.asyncio
    async def test_second_turn_validates_only_its_own_records(self):
        from src.graph.workflow import run_workflow

        llm, prompts = self._llm()
        saver = MemorySaver()

        with (
            patch("src.agents.base.get_llm", return_value=llm),
            patch("src.agents.base.router", _router(llm)),
        ):
            first = await run_workflow(
                "Question one about postgres", "pipeline-turns", llm, checkpointer=saver
            )
            second = await run_workflow(
                "Question two about coffee", "pipeline-turns", llm, checkpointer=saver
            )

        # One finding and one insight per turn, so two checks per turn. Before the
        # fix the second turn re-validated the first turn's records as well: six.
        assert len(first.validation_results) == 2
        assert len(second.validation_results) == 4
        assert all(record["turn"] == first.turn for record in first.validation_results)
        assert [r["turn"] for r in second.validation_results].count(second.turn) == 2

        # Every record says which turn wrote it, and the writer reads this turn's.
        assert all("turn" in finding for finding in second.research_findings)
        assert all("turn" in analysis for analysis in second.analysis_results)
        writer_prompt = [p for p in prompts if "writing agent" in p][-1]
        assert "notes on coffee" in writer_prompt
        assert "notes on postgres" not in writer_prompt
