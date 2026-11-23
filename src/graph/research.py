"""Research workflow: plan questions, fan out finders, chase gaps, then report.

Two waves by default. The planner decides how many questions there are, every
question goes to its own finder running concurrently, a gap analyst looks at what
came back and asks what is still missing, and a second wave closes those gaps
before one report is written.
"""

from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.research import (
    DEFAULT_QUESTIONS,
    MAX_QUESTIONS,
    FinderAgent,
    GapAnalystAgent,
    QuestionPlannerAgent,
    ReportAgent,
)
from src.config import settings
from src.graph.state import AgentState
from src.graph.workflow import _initial_state, checkpointer_scope, reset_thread, resume_info
from src.tools import as_langchain_tools

SECOND_WAVE = 2
"""(kept as a constant so the wave checks read clearly)"""


def create_research_workflow(
    llm: Any = None,
    max_questions: int = DEFAULT_QUESTIONS,
    waves: int = 2,
    checkpointer: Any = None,
) -> Any:
    """Build the research graph: plan -> finders -> gaps -> finders -> report."""
    max_questions = max(1, min(max_questions, MAX_QUESTIONS))
    planner = QuestionPlannerAgent(llm, max_questions=max_questions)
    finder = FinderAgent(llm, tools=as_langchain_tools())
    gaps = GapAnalystAgent(llm)
    reporter = ReportAgent(llm)

    async def plan_node(state: AgentState) -> dict[str, Any]:
        try:
            return await planner.execute(state)
        finally:
            planner._report("done")

    async def finder_node(payload: dict[str, Any]) -> dict[str, Any]:
        return await finder.find(
            index=int(payload["index"]),
            question=str(payload["question"]),
            task=str(payload["task"]),
            turn=str(payload.get("turn", "")),
            wave=int(payload.get("wave", 1)),
        )

    async def gap_node(state: AgentState) -> dict[str, Any]:
        try:
            return await gaps.execute(state)
        finally:
            gaps._report("done")

    async def report_node(state: AgentState) -> dict[str, Any]:
        try:
            return await reporter.execute(state)
        finally:
            reporter._report("done")

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    # Send hands these nodes a question payload, not the graph state.
    graph.add_node("find", finder_node)  # type: ignore[arg-type]
    graph.add_node("gaps", gap_node)
    graph.add_node("refine", finder_node)  # type: ignore[arg-type]
    graph.add_node("report", report_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", dispatch_wave, ["find"])

    if waves >= SECOND_WAVE:
        graph.add_edge("find", "gaps")
        # Either a second wave of finders, or straight to the report.
        graph.add_conditional_edges("gaps", dispatch_gaps, ["refine", "report"])
        graph.add_edge("refine", "report")
    else:
        graph.add_edge("find", "report")

    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer)


def dispatch_wave(state: AgentState) -> list[Send]:
    """One finder per planned question."""
    return [
        Send(
            "find",
            {
                "index": int(item.get("id", index)),
                "question": str(item.get("section") or item.get("task") or state.current_task),
                "task": state.current_task,
                "turn": state.turn,
                "wave": 1,
            },
        )
        for index, item in enumerate(state.plan)
    ]


def dispatch_gaps(state: AgentState) -> list[Send] | str:
    """One finder per gap, or straight to the report when nothing is missing."""
    gaps = [item for item in state.plan if item.get("wave") == SECOND_WAVE]
    if not gaps:
        return "report"
    return [
        Send(
            "refine",
            {
                "index": int(item.get("id", index)),
                "question": str(item.get("section") or state.current_task),
                "task": state.current_task,
                "turn": state.turn,
                "wave": 2,
            },
        )
        for index, item in enumerate(gaps)
    ]


def _max_concurrency() -> int:
    """How many finders may call the model at once."""
    return max(1, min(settings.max_parallel_agents, MAX_QUESTIONS))


async def run_research(  # noqa: PLR0913, PLR0917 - mirrors the other run functions
    query: str,
    thread_id: str = "default",
    llm: Any = None,
    max_questions: int = DEFAULT_QUESTIONS,
    waves: int = 2,
    checkpointer: Any = None,
    fresh: bool = False,
) -> AgentState:
    """Plan, research, fill gaps, and report."""
    config = {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": _max_concurrency(),
    }

    async with checkpointer_scope(checkpointer) as saver:
        if fresh:
            await reset_thread(saver, thread_id)
        app = create_research_workflow(
            llm, max_questions=max_questions, waves=waves, checkpointer=saver
        )
        if not fresh:
            prior = await resume_info(app, config)
            if prior:
                print(f"Resuming thread {thread_id!r} with {prior} earlier messages")
        result = await app.ainvoke(_initial_state(query, thread_id), config=config)

    return result if isinstance(result, AgentState) else AgentState(**result)


async def run_research_stream(  # noqa: PLR0913, PLR0917 - mirrors the other run functions
    query: str,
    thread_id: str = "default",
    llm: Any = None,
    max_questions: int = DEFAULT_QUESTIONS,
    waves: int = 2,
    checkpointer: Any = None,
    fresh: bool = False,
) -> AsyncIterator[AgentState]:
    """Stream a research run, yielding state after every finder lands."""
    config = {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": _max_concurrency(),
    }

    async with checkpointer_scope(checkpointer) as saver:
        if fresh:
            await reset_thread(saver, thread_id)
        app = create_research_workflow(
            llm, max_questions=max_questions, waves=waves, checkpointer=saver
        )
        async for event in app.astream(
            _initial_state(query, thread_id), config=config, stream_mode="values"
        ):
            yield event if isinstance(event, AgentState) else AgentState(**event)
