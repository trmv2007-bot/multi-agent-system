"""Build workflow: outline, research, revise, write, review, fix, hand over.

plan -> research -> replan -> one worker per section -> assemble
     -> review -> (only the flagged sections are rewritten) -> assemble -> report
"""

from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.build import (
    DEFAULT_SECTIONS,
    MAX_SECTIONS,
    AnalysisAgent,
    BuildPlannerAgent,
    CritiqueAgent,
    DesignerAgent,
    PageResearchAgent,
    ReplanAgent,
    SectionWriterAgent,
)
from src.artifacts import assemble_page, page_fragments
from src.config import settings
from src.graph.state import AgentState
from src.graph.workflow import _initial_state, checkpointer_scope, reset_thread, resume_info
from src.tools import as_langchain_tools


def _assemble(state: AgentState) -> str:
    """The page from this turn's sections, in plan order, in its own theme."""
    theme = state.context.get("design_theme") or None
    return assemble_page(state.current_task, page_fragments(state), theme=theme)


def _notes_of(state: AgentState) -> str:
    """Research notes gathered this turn."""
    return "\n".join(
        str(item.get("content", ""))
        for item in state.research_findings
        if item.get("turn") == state.turn
    )


class _Nodes:
    """The build steps, holding the agents they drive."""

    def __init__(self, llm: Any, max_sections: int) -> None:
        self.planner = BuildPlannerAgent(llm, max_sections=max_sections)
        self.researcher = PageResearchAgent(llm, tools=as_langchain_tools())
        self.designer = DesignerAgent(llm, tools=as_langchain_tools())
        self.replanner = ReplanAgent(llm, max_sections=max_sections)
        self.writer = SectionWriterAgent(llm)
        self.critic = CritiqueAgent(llm)
        self.reporter = AnalysisAgent(llm)

    async def plan(self, state: AgentState) -> dict[str, Any]:
        return await self._run(self.planner, state)

    async def research(self, state: AgentState) -> dict[str, Any]:
        return await self._run(self.researcher, state)

    async def design(self, state: AgentState) -> dict[str, Any]:
        return await self._run(self.designer, state)

    async def replan(self, state: AgentState) -> dict[str, Any]:
        return await self._run(self.replanner, state)

    async def assemble(self, state: AgentState) -> dict[str, Any]:
        return {"final_report": _assemble(state)}

    async def review(self, state: AgentState) -> dict[str, Any]:
        return await self._run(self.critic, state)

    async def report(self, state: AgentState) -> dict[str, Any]:
        return await self._run(self.reporter, state)

    async def section(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.writer.write_section(
            index=int(payload["index"]),
            section=str(payload["section"]),
            brief=str(payload["brief"]),
            page_title=str(payload["page_title"]),
            earlier=list(payload.get("earlier") or []),
            turn=str(payload.get("turn", "")),
            notes=str(payload.get("notes", "")),
            design=str(payload.get("design", "")),
        )

    async def fix(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.writer.write_section(
            index=int(payload["index"]),
            section=str(payload["section"]),
            brief=str(payload["brief"]),
            page_title=str(payload["page_title"]),
            earlier=list(payload.get("earlier") or []),
            turn=str(payload.get("turn", "")),
            notes=str(payload.get("notes", "")),
            fix=str(payload.get("fix", "")),
            design=str(payload.get("design", "")),
        )

    @staticmethod
    async def _run(agent: Any, state: AgentState) -> dict[str, Any]:
        """Run one agent, and always leave its status closed out."""
        try:
            update: dict[str, Any] = await agent.execute(state)
        finally:
            agent._report("done")
        return update


def create_build_workflow(
    llm: Any = None,
    max_sections: int = DEFAULT_SECTIONS,
    checkpointer: Any = None,
    review: bool = True,
) -> Any:
    """Build the page graph: plan, research, revise, write, review, report."""
    nodes = _Nodes(llm, max(1, min(max_sections, MAX_SECTIONS)))

    graph = StateGraph(AgentState)
    graph.add_node("plan", nodes.plan)
    graph.add_node("research", nodes.research)
    graph.add_node("design", nodes.design)
    graph.add_node("replan", nodes.replan)
    # Send hands these nodes a payload, not the graph state.
    graph.add_node("sections", nodes.section)  # type: ignore[arg-type]
    graph.add_node("assemble", nodes.assemble)
    graph.add_node("review", nodes.review)
    graph.add_node("fix", nodes.fix)  # type: ignore[arg-type]
    graph.add_node("report", nodes.report)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "design")
    graph.add_edge("design", "replan")
    graph.add_conditional_edges("replan", dispatch_sections, ["sections"])
    graph.add_edge("sections", "assemble")

    if review:
        graph.add_edge("assemble", "review")
        graph.add_conditional_edges("review", dispatch_fixes, ["fix", "report"])
        graph.add_edge("fix", "assemble")
    else:
        graph.add_edge("assemble", "report")

    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)


def _design_of(state: AgentState) -> str:
    """The designer's brief for this turn."""
    return str(state.context.get("design_brief", ""))


def dispatch_sections(state: AgentState) -> list[Send]:
    """One worker per planned section, each carrying the research notes."""
    plan = state.plan or [{"id": 0, "section": "hero", "brief": state.current_task}]
    names = [str(item.get("section", "")) for item in plan]
    notes = _notes_of(state)

    return [
        Send(
            "sections",
            {
                "index": item.get("id", index),
                "section": item.get("section", "section"),
                "brief": item.get("brief", state.current_task),
                "page_title": state.current_task,
                "earlier": names[:index],
                "turn": state.turn,
                "notes": notes,
                "design": _design_of(state),
            },
        )
        for index, item in enumerate(plan)
    ]


MAX_FIX_ROUNDS = 2
"""One review, one repair, one confirmation. More than that and the loop spins."""


def dispatch_fixes(state: AgentState) -> list[Send] | str:
    """Rewrite only the sections the review flagged, at most twice."""
    if int(state.context.get("fix_round", 0)) >= MAX_FIX_ROUNDS:
        return "report"

    findings = [r for r in state.validation_results if r.get("turn") == state.turn]
    if not findings:
        return "report"

    names = [str(item.get("section", "")) for item in state.plan]
    notes = _notes_of(state)
    sends: list[Send] = []

    for finding in findings:
        wanted = str(finding.get("section", "")).strip().lower()
        match = next(
            (
                (index, item)
                for index, item in enumerate(state.plan)
                if wanted and wanted in str(item.get("section", "")).lower()
            ),
            None,
        )
        if match is None:
            match = (0, state.plan[0] if state.plan else {"section": "hero", "brief": ""})
        index, item = match
        sends.append(
            Send(
                "fix",
                {
                    "index": index,
                    "section": item.get("section", "section"),
                    "brief": item.get("brief", ""),
                    "page_title": state.current_task,
                    "earlier": names[:index],
                    "turn": state.turn,
                    "notes": notes,
                    "design": _design_of(state),
                    "fix": str(finding.get("claim", "")),
                },
            )
        )

    return sends or "report"


def _max_concurrency() -> int:
    """How many section workers may call the model at once."""
    return max(1, min(settings.max_parallel_agents, MAX_SECTIONS))


async def run_build(  # noqa: PLR0913, PLR0917 - mirrors the other run functions
    query: str,
    thread_id: str = "default",
    llm: Any = None,
    max_sections: int = DEFAULT_SECTIONS,
    checkpointer: Any = None,
    fresh: bool = False,
    review: bool = True,
) -> AgentState:
    """Outline, research, write, review and hand over one page."""
    config = {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": _max_concurrency(),
    }

    async with checkpointer_scope(checkpointer) as saver:
        if fresh:
            await reset_thread(saver, thread_id)
        app = create_build_workflow(
            llm, max_sections=max_sections, checkpointer=saver, review=review
        )
        if not fresh:
            prior = await resume_info(app, config)
            if prior:
                print(f"Resuming thread {thread_id!r} with {prior} earlier messages")
        result = await app.ainvoke(_initial_state(query, thread_id), config=config)

    return result if isinstance(result, AgentState) else AgentState(**result)


async def run_build_stream(  # noqa: PLR0913, PLR0917 - mirrors the other run functions
    query: str,
    thread_id: str = "default",
    llm: Any = None,
    max_sections: int = DEFAULT_SECTIONS,
    checkpointer: Any = None,
    fresh: bool = False,
    review: bool = True,
) -> AsyncIterator[AgentState]:
    """Stream a build, yielding state after every step."""
    config = {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": _max_concurrency(),
    }

    async with checkpointer_scope(checkpointer) as saver:
        if fresh:
            await reset_thread(saver, thread_id)
        app = create_build_workflow(
            llm, max_sections=max_sections, checkpointer=saver, review=review
        )
        async for event in app.astream(
            _initial_state(query, thread_id), config=config, stream_mode="values"
        ):
            yield event if isinstance(event, AgentState) else AgentState(**event)
