"""Tests for the writer's output contract: measured, re-asked, and trimmed.

Every test here stubs the model, so none of them touch the network.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from src.agents import WriterAgent
from src.agents.contract import (
    WORD_CAP,
    Violation,
    fetched_urls,
    repair_request,
    scaffolding_heading,
    trim_to_cap,
    violation_of,
)
from src.graph.state import AgentState

LONG_BODY = " ".join(["Retrieval adds context that generation alone cannot supply."] * 60)
"""Sixty repetitions: about 600 words against a 180-word cap."""

SCAFFOLDED = "# Executive Summary\n\n" + " ".join(["The system is sound enough."] * 40)


class StubModel:
    """A model that answers from a script and records every prompt."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.prompts: list[list[dict[str, Any]]] = []

    def bind(self, **_kwargs: Any) -> "StubModel":
        return self

    def bind_tools(self, _tools: Any) -> "StubModel":
        return self

    async def ainvoke(self, messages: list[dict[str, Any]], **_kwargs: Any) -> Any:
        self.prompts.append(messages)
        reply = self._replies[min(len(self.prompts) - 1, len(self._replies) - 1)]
        return SimpleNamespace(content=reply, tool_calls=None, response_metadata={})


def state_with(*findings: dict[str, Any], turn: str = "t1") -> AgentState:
    """A state carrying findings the writer may use, all from one turn."""
    state = AgentState(current_task="What is retrieval-augmented generation?", turn=turn)
    for finding in findings:
        state.add_finding(finding)
    return state


def finding(content: str, sources: list[str] | None = None) -> dict[str, Any]:
    """One research finding shaped as the deep-research finder emits it."""
    return {"source": "llm_synthesis", "content": content, "sources": sources or [], "turn": "t1"}


class TestOverLongAnswer:
    """An answer over the cap is re-asked once, then trimmed in code."""

    @pytest.mark.asyncio
    async def test_over_length_reply_is_reasked_once_then_trimmed(self):
        model = StubModel([LONG_BODY, LONG_BODY])
        writer = WriterAgent(llm=model)

        result = await writer.execute(state_with(finding("Retrieval is a technique.")))

        assert len(model.prompts) == 2, "the violation must cost exactly one extra call"
        assert len(result["final_report"].split()) <= WORD_CAP
        assert result["final_report"].startswith("Retrieval adds context")

        reask = model.prompts[1][-1]["content"]
        assert "words" in reask and "under 180 words" in reask

    @pytest.mark.asyncio
    async def test_the_reask_names_the_violation_concretely(self):
        model = StubModel([SCAFFOLDED, SCAFFOLDED])
        writer = WriterAgent(llm=model)

        await writer.execute(state_with(finding("Retrieval is a technique.")))

        reask = model.prompts[1][-1]["content"]
        assert "Executive Summary" in reask
        assert "heading" in reask

    @pytest.mark.asyncio
    async def test_a_compliant_rewrite_is_kept_untouched(self):
        rewritten = "Retrieval-augmented generation adds a search step before the model writes."
        model = StubModel([LONG_BODY, rewritten])
        writer = WriterAgent(llm=model)

        result = await writer.execute(state_with(finding("Retrieval is a technique.")))

        assert result["final_report"] == rewritten


class TestScaffolding:
    """The heading the prompt forbids is gone from the finished answer."""

    @pytest.mark.asyncio
    async def test_scaffolding_heading_is_stripped(self):
        model = StubModel([SCAFFOLDED, SCAFFOLDED])
        writer = WriterAgent(llm=model)

        result = await writer.execute(state_with(finding("Retrieval is a technique.")))

        assert "Executive Summary" not in result["final_report"]
        assert result["final_report"].startswith("The system is sound")

    @pytest.mark.asyncio
    async def test_a_compliant_reply_passes_through_with_no_extra_call(self):
        answer = "Retrieval-augmented generation searches first, then writes the answer."
        model = StubModel([answer])
        writer = WriterAgent(llm=model)

        result = await writer.execute(state_with(finding("Retrieval is a technique.")))

        assert len(model.prompts) == 1
        assert result["final_report"] == answer

    def test_a_sentence_that_starts_with_summary_is_not_a_heading(self):
        answer = "Summary statistics alone do not describe a corpus."

        assert scaffolding_heading(answer) == ""
        assert violation_of(answer) is None

    def test_a_marked_up_heading_is_read_as_one(self):
        assert scaffolding_heading("## Introduction") == "Introduction"
        assert scaffolding_heading("**Key Takeaways**") == "Key Takeaways"
        assert scaffolding_heading("In conclusion: the logs are durable.") == ""


class TestTrim:
    """The deterministic trim never returns more than the cap."""

    def test_a_long_answer_is_cut_at_a_sentence_boundary(self):
        text = " ".join(["Sentences arrive in order here."] * 100)

        trimmed = trim_to_cap(text)

        assert len(trimmed.split()) <= WORD_CAP
        assert trimmed.endswith(".")

    def test_a_word_that_alone_overflows_is_cut_anyway(self):
        trimmed = trim_to_cap(" ".join(["word"] * 400))

        assert len(trimmed.split()) <= WORD_CAP


class TestSourcesFooter:
    """One footer carries the fetched sources and anything unsourced."""

    def test_footer_lists_deduplicated_urls(self):
        findings = [
            finding(
                "one",
                [
                    "https://docs.python.org/3/whatsnew.html",
                    "https://docs.python.org/3/library.html",
                ],
            ),
            finding("two", ["https://www.python.org/downloads/", "https://nodejs.org/en/about"]),
        ]

        assert fetched_urls(findings) == [
            "https://docs.python.org/3/whatsnew.html",
            "https://www.python.org/downloads/",
            "https://nodejs.org/en/about",
        ]

    def test_urls_are_read_out_of_the_fetched_page_text(self):
        findings = [{"source": "web_search", "source_content": "See https://example.com/a."}]

        assert fetched_urls(findings) == ["https://example.com/a"]

    @pytest.mark.asyncio
    async def test_the_answer_ends_with_one_footer_naming_sources_and_unsourced_claims(self):
        model = StubModel(["Acme grew steadily."])
        writer = WriterAgent(llm=model)
        state = state_with(
            finding("Acme grew steadily.", ["https://acme.example/report"]),
            finding("Acme revenue hit $4.2bn.", ["https://news.example/story"]),
        )
        state.add_validation(
            {
                "claim": "Acme revenue hit $4.2bn.",
                "verdict": "unsupported",
                "status": "unverified",
                "turn": "t1",
            }
        )

        result = await writer.execute(state)

        report = result["final_report"]
        assert report.count("Unverified (no fetched source supports these):") == 1
        assert report.count("Sources:") == 1
        assert report.index("Unverified") < report.index("Sources:")
        assert "https://acme.example/report" in report
        assert "$4.2bn" in report


class TestViolationDescription:
    """The re-ask quotes the measurement, not a vague complaint."""

    def test_both_breaches_are_named_together(self):
        violation = Violation(words=962, heading="Executive Summary")

        assert violation.describe() == '962 words with a "Executive Summary" heading'
        assert "under 180 words" in repair_request(violation)
