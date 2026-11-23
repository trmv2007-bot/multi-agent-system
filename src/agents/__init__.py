"""Specialized agents for the research, analysis, validation and writing pipeline."""

import logging
import time
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.agents.base import AgentConfig, BaseAgent
from src.agents.verify import (
    CONFIDENCE_BY_VERDICT,
    CONTRADICTED,
    SEVERITY_BY_VERDICT,
    STATUS_BY_VERDICT,
    UNSUPPORTED,
    claim_key,
    verdicts_from,
    verdicts_of,
    verification_prompt,
)
from src.config import get_tracer
from src.graph.state import AgentRole, AgentState
from src.structured import call_json

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Agent specialized in research and information gathering."""

    def __init__(self, llm: BaseChatModel | None = None, tools: list[BaseTool] | None = None):
        config = AgentConfig(
            role=AgentRole.RESEARCHER,
            name="Researcher",
            description="Gathers information from multiple sources",
            system_prompt="""You are a Research Agent. Your job is to gather accurate, up-to-date information from reliable sources.

Guidelines:
1. Use available tools to search for information
2. Always cite your sources
3. Focus on factual, verifiable information
4. Return findings as structured data
5. If you cannot find information, say so clearly

Output format: Return findings as a list of objects with:
- source: where the information came from
- content: the key information found
- confidence: your confidence level (0-1)
- relevance: how relevant to the query (0-1)""",
            tools=tools or [],
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        with get_tracer().start_as_current_span("agent.researcher.execute") as span:
            span.set_attribute("thread_id", state.thread_id)

            start = time.time()
            query = state.current_task or (state.messages[-1]["content"] if state.messages else "")

            # Build messages with research focus
            messages = self._build_messages(state, f"RESEARCH QUERY: {query}")
            messages.append({"role": "user", "content": f"Research this topic thoroughly: {query}"})

            answer, tool_runs = await self._with_tools_traced(messages)

            findings = [
                {
                    "source": run["name"],
                    "content": run["output"][:1000],
                    # What the source actually returned. Without it the
                    # validator can only see the tool's name, which says
                    # nothing about whether the claim is true.
                    "source_content": run["output"][:6000],
                    "confidence": 0.9,
                    "relevance": 0.8,
                    "turn": state.turn,
                }
                for run in tool_runs
            ]
            if answer.strip():
                findings.append(
                    {
                        "source": "llm_synthesis",
                        "content": answer,
                        # The synthesis has no source of its own: it is
                        # checkable only against everything the tools returned.
                        "source_content": "\n\n".join(run["output"] for run in tool_runs)[:6000],
                        "confidence": 0.7,
                        "relevance": 0.9,
                        "turn": state.turn,
                    }
                )

            for finding in findings:
                finding["timestamp"] = datetime.now(UTC).isoformat()

            return {
                "research_findings": findings,
                "agent_timings": {self.config.role.value: time.time() - start},
                "messages": [self._message(f"Research completed: {len(findings)!r} findings")],
            }


class AnalystAgent(BaseAgent):
    """Agent specialized in analysis and synthesis."""

    def __init__(self, llm: BaseChatModel | None = None, tools: list[BaseTool] | None = None):
        config = AgentConfig(
            role=AgentRole.ANALYST,
            name="Analyst",
            description="Analyzes research findings and extracts insights",
            system_prompt="""You are an Analysis Agent. Your job is to analyze research findings and extract key insights, patterns, and implications.

Guidelines:
1. Synthesize information from multiple sources
2. Identify patterns, trends, and contradictions
3. Provide structured analysis with clear reasoning
4. Say which findings each conclusion rests on
5. Highlight areas needing further research

Output format: Return analysis as objects with:
- topic: what was analyzed
- insight: key insight or conclusion
- evidence: supporting evidence from findings
- implications: what this means""",
            tools=tools or [],
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        with get_tracer().start_as_current_span("agent.analyst.execute") as span:
            span.set_attribute("thread_id", state.thread_id)

            start = time.time()

            if not state.research_findings:
                return {
                    "analysis_results": [
                        {
                            "topic": "No findings to analyze",
                            "insight": "No research findings available for analysis",
                            "evidence": [],
                            "implications": "Research phase must complete first",
                            "turn": state.turn,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    ],
                    "agent_timings": {self.config.role.value: time.time() - start},
                }

            # Build analysis prompt
            findings_summary = "\n".join(
                [f"[{f['source']}] {f['content'][:300]}" for f in state.research_findings]
            )

            messages = self._build_messages(state, f"FINDINGS TO ANALYZE:\n{findings_summary}")
            messages.append(
                {
                    "role": "user",
                    "content": "Analyze these findings and extract key insights, patterns, and implications.",
                }
            )

            response = await self._ainvoke(messages)

            # Parse response into structured analysis
            analyses = [
                {
                    "topic": state.current_task or "General analysis",
                    "insight": response.content,
                    "evidence": [f["source"] for f in state.research_findings],
                    "implications": "See detailed analysis above",
                    "turn": state.turn,
                }
            ]

            for analysis in analyses:
                analysis["timestamp"] = datetime.now(UTC).isoformat()

            return {
                "analysis_results": analyses,
                "agent_timings": {self.config.role.value: time.time() - start},
                "messages": [self._message(f"Analysis completed: {len(analyses)} insights")],
            }


class WriterAgent(BaseAgent):
    """Agent specialized in writing and report generation."""

    def __init__(self, llm: BaseChatModel | None = None, tools: list[BaseTool] | None = None):
        config = AgentConfig(
            role=AgentRole.WRITER,
            name="Writer",
            description="Generates final reports from analysis",
            system_prompt="""You are the writing agent. You turn research and analysis into the answer.

Guidelines:
1. Answer in the first sentence
2. Default to a short reply: at most 180 words. No "Executive Summary", no
   "Conclusions & Recommendations" scaffolding unless the user asked for a report
3. Cite sources as short links, only where a claim needs one
4. Say plainly when the evidence was thin rather than padding the answer
5. Expand into a full report only when the request asks for a document

Output: plain prose or a few bullets. Short. Never paste HTML, code, or a whole
web page into the answer — describe it in one line instead.""",
            tools=tools or [],
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        with get_tracer().start_as_current_span("agent.writer.execute") as span:
            span.set_attribute("thread_id", state.thread_id)

            start = time.time()

            # The validator's grade decides what the writer may treat as fact:
            # contradicted findings never reach the model, unsupported ones
            # reach it labelled.
            verdicts = verdicts_of(state.validation_results, state.turn)
            kept: list[dict[str, Any]] = []
            unverified: list[dict[str, Any]] = []
            dropped = 0
            for finding in state.research_findings:
                verdict = verdicts.get(claim_key(finding), "")
                if verdict == CONTRADICTED:
                    dropped += 1
                    continue
                kept.append(finding)
                if verdict == UNSUPPORTED:
                    unverified.append(finding)
            unverified_keys = {claim_key(finding) for finding in unverified}

            findings_str = "\n".join(
                f"- [{finding['source']}] {finding['content'][:200]}"
                + (
                    "  (UNVERIFIED: no fetched source supports this)"
                    if claim_key(finding) in unverified_keys
                    else ""
                )
                for finding in kept
            )

            kept_analysis = [
                analysis
                for analysis in state.analysis_results
                if verdicts.get(claim_key(analysis), "") != CONTRADICTED
            ]
            dropped += len(state.analysis_results) - len(kept_analysis)
            analysis_str = "\n".join(f"- {a['insight'][:200]}" for a in kept_analysis)

            dropped_note = (
                f"\n\n{dropped} claim(s) were removed because their sources contradict them. "
                "Do not restate them or invent replacements for them."
                if dropped
                else ""
            )

            messages = self._build_messages(state, findings=kept)
            messages.append(
                {
                    "role": "user",
                    "content": f"""Create a comprehensive report based on:

RESEARCH FINDINGS:
{findings_str}

ANALYSIS:
{analysis_str}{dropped_note}

Treat anything labelled UNVERIFIED as unconfirmed rather than fact.

Original query: {state.current_task or state.messages[0]["content"] if state.messages else "Unknown"}""",
                }
            )

            response = await self._ainvoke(messages)

            report = self._text_of(response).rstrip()
            note = self._unverified_note(unverified, dropped)
            if note:
                report = f"{report}\n\n{note}"

            return {
                "final_report": report,
                "agent_timings": {self.config.role.value: time.time() - start},
                "messages": [self._message("Report generated")],
            }

    @staticmethod
    def _unverified_note(unverified: list[dict[str, Any]], dropped: int) -> str:
        """The provenance footer naming what the sources could not support."""
        lines: list[str] = []
        if unverified:
            lines.append("Unverified (no fetched source supports these):")
            lines.extend(
                f"- {claim_key(finding)[:120]} (from {finding.get('source', 'unknown')})"
                for finding in unverified[:5]
            )
        if dropped:
            lines.append(f"Removed {dropped} claim(s) that their sources contradicted.")
        return "\n".join(lines)


class ValidatorAgent(BaseAgent):
    """
    Grades claims against the source text fetched for them.

    One model call per pass, built by
    :func:`src.agents.verify.verification_prompt`. ``scope`` selects the pass:
    ``findings`` runs in parallel with the analyst and grades the research
    findings against their fetched sources, ``analysis`` runs after it and grades
    the analyst's insights against the findings, and ``all`` does both for direct
    callers.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        tools: list[BaseTool] | None = None,
        scope: str = "all",
    ):
        if scope not in ("findings", "analysis", "all"):
            raise ValueError(f"Unknown validator scope: {scope}")
        self.scope = scope
        config = AgentConfig(
            role=AgentRole.VALIDATOR,
            name="Validator",
            description="Grades claims against the source text fetched for them",
            system_prompt="""You verify claims against source text.

Rules:
1. Only the source text supplied with a claim counts as evidence
2. Mark a claim supported only when that source states it
3. Mark it contradicted when the source says something incompatible with it
4. Mark it unsupported when the source is silent, off-topic or absent
5. Never use your own knowledge to fill a gap in the source

Output: JSON only, one entry per claim, no prose.""",
            tools=tools or [],
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        with get_tracer().start_as_current_span(f"agent.validator.{self.scope}.execute") as span:
            span.set_attribute("thread_id", state.thread_id)

            start = time.time()

            validations: list[dict[str, Any]] = []
            if self.scope in ("findings", "all"):
                validations.extend(await self._validate_findings(state))
            if self.scope in ("analysis", "all"):
                validations.extend(await self._validate_analysis(state))

            stamp = datetime.now(UTC).isoformat()
            for validation in validations:
                validation["timestamp"] = stamp

            return {
                "validation_results": validations,
                "agent_timings": {f"validator.{self.scope}": time.time() - start},
                "messages": [self._message(f"Validation completed: {len(validations)} checks")],
            }

    async def _validate_findings(self, state: AgentState) -> list[dict[str, Any]]:
        """Grade this turn's findings against the text their sources returned."""
        findings = [finding for finding in state.research_findings if _in_turn(finding, state.turn)]
        if not findings:
            return []

        claims = [
            {
                "claim": str(finding.get("content", "")),
                "source": str(finding.get("source", "unknown")),
                "source_content": str(finding.get("source_content", "")),
            }
            for finding in findings
        ]
        slots = await self._grade(claims)

        return [
            _validation_record(
                claim=claim_key(finding),
                slot=slot,
                evidence=[str(finding.get("source", "unknown"))],
                scope="findings",
                turn=state.turn,
            )
            for finding, slot in zip(findings, slots, strict=True)
        ]

    async def _validate_analysis(self, state: AgentState) -> list[dict[str, Any]]:
        """Grade this turn's insights against the findings they rest on."""
        analyses = [
            analysis for analysis in state.analysis_results if _in_turn(analysis, state.turn)
        ]
        if not analyses:
            return []

        evidence = "\n".join(
            f"[{finding.get('source', 'unknown')}] {str(finding.get('content', ''))[:500]}"
            for finding in state.research_findings
        )
        claims = [
            {
                "claim": str(analysis.get("insight", "")),
                "source": "research findings",
                "source_content": evidence,
            }
            for analysis in analyses
        ]
        slots = await self._grade(claims)

        return [
            _validation_record(
                claim=claim_key(analysis),
                slot=slot,
                evidence=list(analysis.get("evidence") or []),
                scope="analysis",
                turn=state.turn,
            )
            for analysis, slot in zip(analyses, slots, strict=True)
        ]

    async def _grade(self, claims: list[dict[str, Any]]) -> list[dict[str, str] | None]:
        """Ask the model for one verdict per claim.

        A call that fails, or a reply the reader cannot make sense of even after
        a repair and a retry, grades nothing as supported: every slot comes back
        ``None`` and the caller records those claims as unverified.
        """
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": verification_prompt(claims)},
        ]
        try:
            payload = await call_json(
                self._json_sender(self._actor()),
                messages,
                site="verify.grade",
                expect=(list, dict),
            )
        except Exception as error:
            logger.warning("verification unavailable, grading nothing as supported: %s", error)
            self._report("verification unavailable", self._actor())
            return [None] * len(claims)
        return verdicts_from([payload], len(claims))


def _in_turn(item: dict[str, Any], turn: str) -> bool:
    """Whether an item belongs to the turn being validated.

    Items written before turns existed carry none and are graded anyway.
    """
    item_turn = item.get("turn")
    return not item_turn or item_turn == turn


def _validation_record(
    claim: str,
    slot: dict[str, str] | None,
    evidence: list[str],
    scope: str,
    turn: str,
) -> dict[str, Any]:
    """Turn one verdict into the record the writer and the graph read."""
    if slot is None:
        # An empty or unreadable reply is a failed check, never a pass.
        verdict, reason = UNSUPPORTED, "the verifier returned no verdict for this claim"
    else:
        verdict, reason = slot["verdict"], slot["reason"]

    return {
        "claim": claim,
        "verdict": verdict,
        "status": STATUS_BY_VERDICT[verdict],
        "reason": reason or "no reason given",
        "evidence": evidence,
        "confidence": CONFIDENCE_BY_VERDICT[verdict],
        "severity": SEVERITY_BY_VERDICT[verdict],
        "scope": scope,
        "turn": turn,
    }


class OrchestratorAgent(BaseAgent):
    """Orchestrator that coordinates the workflow."""

    def __init__(self, llm: BaseChatModel | None = None):
        config = AgentConfig(
            role=AgentRole.ORCHESTRATOR,
            name="Orchestrator",
            description="Coordinates the multi-agent workflow",
            system_prompt="""You are the Orchestrator. Your job is to coordinate the workflow and make routing decisions.

Guidelines:
1. Analyze the current state
2. Decide which agent should run next
3. Determine if parallel execution is possible
4. Check for completion or errors
5. Provide clear next steps

Output: Return the next agent role(s) to execute.""",
            tools=[],
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        """Orchestrator doesn't modify state directly - it's handled by the graph."""
        return {}

    def decide_next(self, state: AgentState) -> list[AgentRole]:
        """Decide which agents should run next based on state."""
        # Simple decision logic - in production this would be more sophisticated
        if not state.research_findings:
            return [AgentRole.RESEARCHER]
        elif not state.analysis_results:
            return [AgentRole.ANALYST]
        elif not state.validation_results:
            return [AgentRole.VALIDATOR]
        elif not state.final_report:
            return [AgentRole.WRITER]
        else:
            return []


# Factory function to create agents
def create_agents(
    llm: BaseChatModel | None = None, tools: list[BaseTool] | None = None
) -> dict[AgentRole, BaseAgent]:
    """Create all agents for the system."""
    return {
        AgentRole.RESEARCHER: ResearcherAgent(llm, tools),
        AgentRole.ANALYST: AnalystAgent(llm, tools),
        AgentRole.WRITER: WriterAgent(llm, tools),
        AgentRole.VALIDATOR: ValidatorAgent(llm, tools),
        AgentRole.ORCHESTRATOR: OrchestratorAgent(llm),
    }
