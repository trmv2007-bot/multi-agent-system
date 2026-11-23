"""Tests for the verification layer: real grading, and the grade driving the run.

Every test here stubs the model, so none of them touch the network.
"""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.agents import ValidatorAgent, WriterAgent
from src.agents.verify import (
    CONTRADICTED,
    SUPPORTED,
    UNSUPPORTED,
    parse_verdicts,
    verdicts_of,
)
from src.graph.state import AgentState
from src.graph.workflow import _initial_state, create_workflow

FABRICATED = "Acme revenue hit $4.2bn in March 2026."
FETCHED = "Acme reported revenue of $1.1bn for the 2025 financial year."
SUPPORTED_CLAIM = "Acme grew headcount 12% in 2025."


class StubModel:
    """A model that answers from a routing function and records every prompt."""

    def __init__(self, respond: Any):
        self._respond = respond
        self.prompts: list[list[dict[str, Any]]] = []
        self.binds: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any) -> "StubModel":
        self.binds.append(kwargs)
        return self

    def bind_tools(self, _tools: Any) -> "StubModel":
        return self

    async def ainvoke(self, messages: list[dict[str, Any]], **_kwargs: Any) -> Any:
        self.prompts.append(messages)
        return SimpleNamespace(
            content=self._respond(messages), tool_calls=None, response_metadata={}
        )


def verdict_reply(*verdicts: str) -> str:
    """The JSON a well-behaved verifier returns for one verdict per claim."""
    return json.dumps(
        [
            {"index": index, "verdict": verdict, "reason": f"because {verdict}"}
            for index, verdict in enumerate(verdicts)
        ]
    )


def is_verdict_prompt(messages: list[dict[str, Any]]) -> bool:
    """Whether this call is the validator asking for verdicts."""
    return any("Check every claim below" in str(step.get("content", "")) for step in messages)


def is_writer_prompt(messages: list[dict[str, Any]]) -> bool:
    """Whether this call is the writer composing the answer."""
    return any("Create a comprehensive report" in str(step.get("content", "")) for step in messages)


def finding(content: str, source_content: str = "") -> dict[str, Any]:
    """One research finding shaped as the researcher emits it."""
    return {"source": "web_search", "content": content, "source_content": source_content}


class TestFindingsVerification:
    """The validator grades claims against the source text actually fetched."""

    @pytest.mark.asyncio
    async def test_fabricated_finding_is_graded_unsupported(self):
        model = StubModel(lambda _messages: verdict_reply(UNSUPPORTED))
        validator = ValidatorAgent(llm=model, scope="findings")
        state = AgentState(current_task="Acme revenue")
        state.add_finding(finding(FABRICATED, FETCHED))

        result = await validator.execute(state)

        [record] = result["validation_results"]
        assert record["verdict"] == UNSUPPORTED
        assert record["status"] == "unverified"
        prompt = model.prompts[0][-1]["content"]
        assert FETCHED in prompt, "the prompt must carry the fetched text, not just the tool name"
        assert FABRICATED in prompt

    @pytest.mark.asyncio
    async def test_supported_finding_is_graded_verified(self):
        model = StubModel(lambda _messages: verdict_reply(SUPPORTED))
        validator = ValidatorAgent(llm=model, scope="findings")
        state = AgentState(current_task="Acme headcount")
        state.add_finding(finding(SUPPORTED_CLAIM, "Headcount grew from 40 to 45, up 12%."))

        result = await validator.execute(state)

        [record] = result["validation_results"]
        assert record["verdict"] == SUPPORTED
        assert record["status"] == "verified"
        assert record["evidence"] == ["web_search"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reply", ["", None, "I could not tell you.", "{not json"])
    async def test_an_unreadable_reply_never_becomes_a_pass(self, reply):
        model = StubModel(lambda _messages: reply)
        validator = ValidatorAgent(llm=model, scope="findings")
        state = AgentState(current_task="Acme headcount")
        state.add_finding(finding(SUPPORTED_CLAIM, "Headcount grew 12%."))

        result = await validator.execute(state)

        [record] = result["validation_results"]
        assert record["status"] != "verified"
        assert record["verdict"] == UNSUPPORTED
        assert "no verdict" in record["reason"]

    @pytest.mark.asyncio
    async def test_a_missing_verdict_for_one_claim_does_not_cover_the_rest(self):
        reply = verdict_reply(SUPPORTED)
        model = StubModel(lambda _messages: reply)
        validator = ValidatorAgent(llm=model, scope="findings")
        state = AgentState(current_task="two claims")
        state.add_finding(finding("first claim", "first source"))
        state.add_finding(finding("second claim", "second source"))

        result = await validator.execute(state)

        statuses = [record["status"] for record in result["validation_results"]]
        assert statuses == ["verified", "unverified"]

    @pytest.mark.asyncio
    async def test_a_failed_verifier_call_grades_nothing_as_supported(self):
        class DeadModel(StubModel):
            async def ainvoke(self, _messages: list[dict[str, Any]], **_kwargs: Any) -> Any:
                raise RuntimeError("gateway unreachable")

        validator = ValidatorAgent(llm=DeadModel(lambda _messages: ""), scope="findings")
        state = AgentState(current_task="Acme headcount")
        state.add_finding(finding(FABRICATED, FETCHED))

        result = await validator.execute(state)

        [record] = result["validation_results"]
        assert record["status"] == "unverified"


class TestAnalysisVerification:
    """The analysis gate is a judgement about the evidence, not a constant."""

    @pytest.mark.asyncio
    async def test_analysis_is_graded_against_the_findings(self):
        model = StubModel(lambda _messages: verdict_reply(UNSUPPORTED))
        validator = ValidatorAgent(llm=model, scope="analysis")
        state = AgentState(current_task="Acme revenue")
        state.add_finding(finding("Revenue was $1.1bn in 2025.", "Revenue was $1.1bn in 2025."))
        state.add_analysis({"insight": FABRICATED, "evidence": ["web_search"]})

        result = await validator.execute(state)

        [record] = result["validation_results"]
        assert record["scope"] == "analysis"
        assert record["status"] == "unverified"
        prompt = model.prompts[0][-1]["content"]
        assert "Revenue was $1.1bn in 2025." in prompt
        assert FABRICATED in prompt

    @pytest.mark.asyncio
    async def test_analysis_has_no_self_reported_confidence_to_pass_with(self):
        from src.agents import AnalystAgent

        model = StubModel(lambda _messages: "Acme is growing steadily.")
        analyst = AnalystAgent(llm=model)
        state = AgentState(current_task="Acme")
        state.add_finding(finding("Revenue was $1.1bn.", "Revenue was $1.1bn."))

        result = await analyst.execute(state)

        [analysis] = result["analysis_results"]
        assert "confidence" not in analysis


class TestWriterConsumesTheGrade:
    """The writer sees the grade: contradicted claims never reach it."""

    @pytest.mark.asyncio
    async def test_contradicted_finding_is_dropped_from_the_writers_input(self):
        model = StubModel(lambda _messages: "Acme grew.")
        writer = WriterAgent(llm=model)
        state = AgentState(current_task="Acme", turn="t1")
        state.add_finding(finding(SUPPORTED_CLAIM, "Headcount grew 12%."))
        state.add_finding(finding(FABRICATED, FETCHED))
        state.add_analysis({"insight": FABRICATED, "evidence": ["web_search"], "turn": "t1"})
        state.add_validation(
            {"claim": SUPPORTED_CLAIM, "verdict": SUPPORTED, "status": "verified", "turn": "t1"}
        )
        state.add_validation(
            {"claim": FABRICATED, "verdict": CONTRADICTED, "status": "contradicted", "turn": "t1"}
        )

        result = await writer.execute(state)

        sent = "\n".join(str(step.get("content", "")) for step in model.prompts[0])
        assert SUPPORTED_CLAIM in sent
        assert FABRICATED not in sent
        assert "Removed 2 claim(s)" in result["final_report"]

    @pytest.mark.asyncio
    async def test_unsupported_finding_is_flagged_in_the_prompt_and_the_report(self):
        model = StubModel(lambda _messages: "Acme is doing well.")
        writer = WriterAgent(llm=model)
        state = AgentState(current_task="Acme", turn="t1")
        state.add_finding(finding(FABRICATED, FETCHED))
        state.add_validation(
            {"claim": FABRICATED, "verdict": UNSUPPORTED, "status": "unverified", "turn": "t1"}
        )

        result = await writer.execute(state)

        sent = "\n".join(str(step.get("content", "")) for step in model.prompts[0])
        assert "UNVERIFIED" in sent
        assert "Unverified (no fetched source supports these):" in result["final_report"]
        assert FABRICATED in result["final_report"]

    @pytest.mark.asyncio
    async def test_a_graded_supported_finding_adds_no_warning(self):
        model = StubModel(lambda _messages: "Acme grew 12%.")
        writer = WriterAgent(llm=model)
        state = AgentState(current_task="Acme", turn="t1")
        state.add_finding(finding(SUPPORTED_CLAIM, "Headcount grew 12%."))
        state.add_validation(
            {"claim": SUPPORTED_CLAIM, "verdict": SUPPORTED, "status": "verified", "turn": "t1"}
        )

        result = await writer.execute(state)

        assert result["final_report"] == "Acme grew 12%."


class TestGraphBranchesOnTheGrade:
    """A fully contradicted run does not reach the writer."""

    @pytest.mark.asyncio
    async def test_contradicted_run_abstains_instead_of_writing(self):
        model = StubModel(
            lambda messages: (
                verdict_reply(CONTRADICTED)
                if is_verdict_prompt(messages)
                else "Acme revenue was $4.2bn in March 2026."
            )
        )
        app = create_workflow(llm=model)

        result = await app.ainvoke(
            _initial_state("What was Acme's revenue?", "contradicted"),
            config={"configurable": {"thread_id": "contradicted"}},
        )
        state = result if isinstance(result, AgentState) else AgentState(**result)

        assert state.context.get("verification") == "abstained"
        assert "could not verify the evidence" in state.final_report
        assert not any(is_writer_prompt(prompt) for prompt in model.prompts)

    @pytest.mark.asyncio
    async def test_supported_run_reaches_the_writer(self):
        model = StubModel(
            lambda messages: (
                verdict_reply(SUPPORTED)
                if is_verdict_prompt(messages)
                else "Acme revenue was $1.1bn in 2025."
            )
        )
        app = create_workflow(llm=model)

        result = await app.ainvoke(
            _initial_state("What was Acme's revenue?", "supported"),
            config={"configurable": {"thread_id": "supported"}},
        )
        state = result if isinstance(result, AgentState) else AgentState(**result)

        assert any(is_writer_prompt(prompt) for prompt in model.prompts)
        assert state.context.get("verification") != "abstained"
        assert state.final_report


class TestVerdictParsing:
    """The parser is strict about what counts as a verdict."""

    def test_reads_a_fenced_json_block(self):
        reply = 'Here you go:\n```json\n[{"index": 0, "verdict": "supported", "reason": "ok"}]\n```'
        assert parse_verdicts(reply, 1) == [{"verdict": SUPPORTED, "reason": "ok"}]

    def test_an_unknown_verdict_is_no_verdict(self):
        reply = json.dumps([{"index": 0, "verdict": "probably fine", "reason": "vibes"}])
        assert parse_verdicts(reply, 1) == [None]

    def test_an_out_of_range_index_is_ignored(self):
        reply = json.dumps([{"index": 4, "verdict": SUPPORTED, "reason": "wrong claim"}])
        assert parse_verdicts(reply, 1) == [None]

    def test_verdicts_are_scoped_to_a_turn(self):
        records = [
            {"claim": "a", "verdict": SUPPORTED, "turn": "old"},
            {"claim": "b", "verdict": CONTRADICTED, "turn": "new"},
        ]
        assert verdicts_of(records, turn="new") == {"b": CONTRADICTED}
        assert verdicts_of(records) == {"a": SUPPORTED, "b": CONTRADICTED}

    def test_finds_the_json_after_prose_with_brackets(self):
        reply = (
            "Checking [the claim] now:\n"
            '[{"index": 0, "verdict": "contradicted", "reason": "the source says otherwise"}]'
        )
        assert parse_verdicts(reply, 1) == [
            {"verdict": CONTRADICTED, "reason": "the source says otherwise"}
        ]

    def test_a_lone_claim_may_omit_the_index(self):
        reply = '{"verdict": "supported", "reason": "the source states it"}'
        assert parse_verdicts(reply, 1) == [
            {"verdict": SUPPORTED, "reason": "the source states it"}
        ]

    def test_prose_only_is_no_verdict(self):
        assert parse_verdicts("It all looks fine to me.", 2) == [None, None]

    def test_reads_a_reply_that_dropped_the_quotes(self):
        # Observed from a pooled model: the reason value is not quoted, so the
        # reply is not JSON, but every verdict in it is unambiguous.
        reply = (
            '[{"index": 0, "verdict": "supported", "reason": Source text states it is 389,444.}, '
            '{"index": 1, "verdict": "contradicted", "reason": Source says otherwise."}]'
        )
        assert parse_verdicts(reply, 2) == [
            {"verdict": SUPPORTED, "reason": "Source text states it is 389,444."},
            {"verdict": CONTRADICTED, "reason": "Source says otherwise."},
        ]
