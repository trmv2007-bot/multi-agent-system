"""A review repair must replace the section it fixed, not join it on the page.

The live preview keys fragments by section index, so a repaired section looks
right while the run is streaming. The page written at the end was built by
joining every result, which kept the broken section and its fix side by side.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.state import AgentState

THEME = json.dumps(
    {
        "feeling": "warm",
        "palette": {
            "bg": "#14100c",
            "surface": "#1e1712",
            "accent": "#e8a33d",
            "text": "#f7f0e6",
            "muted": "#b4a08a",
        },
        "radius": "10px",
    }
)


def _router(llm):
    """A provider chain with one provider, so no test reaches the network."""
    stub = MagicMock()
    stub.provider_names.return_value = ["stub"]
    stub.next_provider.return_value = llm
    return stub


class TestBuildSectionRepair:
    """One page, one section per index, however many rewrites it took."""

    def test_repeated_index_keeps_the_latest_fragment(self):
        from src.artifacts import page_fragments

        state = AgentState(turn="t1")
        state.subtask_results = [
            {"index": 0, "content": "<section class='hero'>Hero</section>", "turn": "t1"},
            {"index": 1, "content": "<section class='cta'></section>", "turn": "t1"},
            {"index": 1, "content": "<section class='cta'>Book now</section>", "turn": "t1"},
        ]

        assert page_fragments(state) == [
            "<section class='hero'>Hero</section>",
            "<section class='cta'>Book now</section>",
        ]

    @pytest.mark.asyncio
    async def test_repaired_section_replaces_the_broken_one(self):
        from src.graph.build import run_build

        plan = json.dumps(
            [{"section": "hero", "brief": "headline"}, {"section": "cta", "brief": "closing"}]
        )
        review = json.dumps([{"section": "cta", "issue": "the button text is empty"}])
        queue = [
            plan,
            "notes",
            THEME,
            plan,
            '<section class="hero"><h1>Hero</h1></section>',
            '<section class="cta"><h2></h2></section>',  # broken
            review,
            '<section class="cta"><h2>Book now</h2></section>',  # fixed
            "[]",  # confirmation round finds nothing left
            "Fixed the button.",
        ]

        async def ainvoke(messages, **kwargs):
            return MagicMock(
                content=queue.pop(0) if queue else "", tool_calls=None, usage_metadata={}
            )

        llm = AsyncMock()
        llm.ainvoke = ainvoke
        llm.bind_tools = MagicMock(return_value=llm)

        with (
            patch("src.agents.base.get_llm", return_value=llm),
            patch("src.agents.base.router", _router(llm)),
        ):
            state = await run_build("A page", "build-repair", llm)

        page = state.final_report
        assert page.count('class="cta"') == 1
        assert "<h2></h2>" not in page
        assert "Book now" in page
        # The results still record both writes, so the history of the fix survives.
        assert [r["index"] for r in state.subtask_results] == [0, 1, 1]
