"""Specialized agents for the research, analysis, validation and writing pipeline."""

import time
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.agents.base import VERIFIED_CONFIDENCE, AgentConfig, BaseAgent
from src.config import get_tracer
from src.graph.state import AgentRole, AgentState


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
                    "confidence": 0.9,
                    "relevance": 0.8,
                }
                for run in tool_runs
            ]
            if answer.strip():
                findings.append(
                    {
                        "source": "llm_synthesis",
                        "content": answer,
                        "confidence": 0.7,
                        "relevance": 0.9,
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
4. Rate confidence in your conclusions
5. Highlight areas needing further research

Output format: Return analysis as objects with:
- topic: what was analyzed
- insight: key insight or conclusion
- evidence: supporting evidence from findings
- confidence: confidence level (0-1)
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
                            "confidence": 0.0,
                            "implications": "Research phase must complete first",
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
                    "confidence": 0.8,
                    "implications": "See detailed analysis above",
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

            # Build context from all previous work
            findings_str = "\n".join(
                [f"- [{f['source']}] {f['content'][:200]}" for f in state.research_findings]
            )

            analysis_str = "\n".join([f"- {a['insight'][:200]}" for a in state.analysis_results])

            messages = self._build_messages(state)
            messages.append(
                {
                    "role": "user",
                    "content": f"""Create a comprehensive report based on:

RESEARCH FINDINGS:
{findings_str}

ANALYSIS:
{analysis_str}

Original query: {state.current_task or state.messages[0]["content"] if state.messages else "Unknown"}""",
                }
            )

            response = await self._ainvoke(messages)

            return {
                "final_report": response.content,
                "agent_timings": {self.config.role.value: time.time() - start},
                "messages": [self._message("Report generated")],
            }


class ValidatorAgent(BaseAgent):
    """
    Agent specialized in validation and fact-checking.

    ``scope`` selects what this instance checks: ``findings`` runs in parallel
    with the analyst, ``analysis`` has to run after it, and ``all`` does both
    for direct callers.
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
            description="Validates findings and checks for hallucinations",
            system_prompt="""You are a Validation Agent. Your job is to verify the accuracy of research findings and analysis.

Guidelines:
1. Check claims against known facts
2. Identify potential hallucinations or fabrications
3. Verify sources are credible
4. Flag unsupported claims
5. Rate overall reliability

Output format: Return validation results as objects with:
- claim: the claim being validated
- status: verified / unverified / contradicted / hallucinated
- evidence: supporting or contradicting evidence
- confidence: confidence in validation (0-1)
- severity: low / medium / high / critical""",
            tools=tools or [],
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        with get_tracer().start_as_current_span(f"agent.validator.{self.scope}.execute") as span:
            span.set_attribute("thread_id", state.thread_id)

            start = time.time()

            validations: list[dict[str, Any]] = []
            if self.scope in ("findings", "all"):
                validations.extend(self._validate_findings(state.research_findings))
            if self.scope in ("analysis", "all"):
                validations.extend(self._validate_analysis(state.analysis_results))

            stamp = datetime.now(UTC).isoformat()
            for validation in validations:
                validation["timestamp"] = stamp

            return {
                "validation_results": validations,
                "agent_timings": {f"validator.{self.scope}": time.time() - start},
                "messages": [self._message(f"Validation completed: {len(validations)} checks")],
            }

    @staticmethod
    def _validate_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check research findings for hallucination markers and unsourced claims."""
        hallucination_indicators = [
            "according to my knowledge cutoff",
            "as an ai language model",
            "i cannot access",
            "i don't have access",
        ]

        validations = []
        for finding in findings:
            content = finding.get("content", "")
            source = finding.get("source", "")

            if any(indicator in content.lower() for indicator in hallucination_indicators):
                status, severity = "hallucinated", "high"
            elif source in ("llm_synthesis", "unknown"):
                status, severity = "unverified", "medium"
            else:
                status, severity = "verified", "low"

            validations.append(
                {
                    "claim": content[:200],
                    "status": status,
                    "evidence": [source],
                    "confidence": 0.8 if status == "verified" else 0.4,
                    "severity": severity,
                }
            )
        return validations

    @staticmethod
    def _validate_analysis(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check that each analysis insight carries its supporting confidence."""
        return [
            {
                "claim": analysis.get("insight", "")[:200],
                "status": (
                    "verified"
                    if analysis.get("confidence", 0) > VERIFIED_CONFIDENCE
                    else "unverified"
                ),
                "evidence": analysis.get("evidence", []),
                "confidence": analysis.get("confidence", 0.5),
                "severity": "low",
            }
            for analysis in analyses
        ]


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
