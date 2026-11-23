"""Team workflow: one orchestrator fanning out to parallel worker subagents."""

from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.team import (
    DEFAULT_MAX_SUBTASKS,
    MAX_SUBTASKS,
    PlannerAgent,
    SynthesizerAgent,
    WorkerAgent,
)
from src.config import settings
from src.graph.state import AgentState
from src.graph.workflow import (
    _initial_state,
    checkpointer_scope,
    reset_thread,
    resume_info,
)
from src.tools import as_langchain_tools


def create_team_workflow(
    llm: Any = None,
    max_subtasks: int = DEFAULT_MAX_SUBTASKS,
    checkpointer: Any = None,
) -> Any:
    """
    Build the team graph.

    plan -> worker (one concurrent Send per subtask) -> synthesize
    """
    max_subtasks = max(1, min(max_subtasks, MAX_SUBTASKS))
    planner = PlannerAgent(llm, max_subtasks=max_subtasks)
    worker = WorkerAgent(llm, tools=as_langchain_tools())
    synthesizer = SynthesizerAgent(llm)

    async def plan_node(state: AgentState) -> dict[str, Any]:
        try:
            return await planner.execute(state)
        finally:
            planner._report("done")

    async def worker_node(payload: dict[str, Any]) -> dict[str, Any]:
        return await worker.run_subtask(
            index=int(payload["index"]),
            task=str(payload["task"]),
            query=str(payload["query"]),
            turn=str(payload.get("turn", "")),
        )

    async def synthesize_node(state: AgentState) -> dict[str, Any]:
        try:
            return await synthesizer.execute(state)
        finally:
            synthesizer._report("done")

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    # Send passes the subtask payload, not the graph state, to this node.
    graph.add_node("worker", worker_node)  # type: ignore[arg-type]
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", dispatch_workers, ["worker"])
    graph.add_edge("worker", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=checkpointer)


def dispatch_workers(state: AgentState) -> list[Send]:
    """Fan the plan out into one concurrent worker per subtask."""
    plan = state.plan or [{"id": 0, "task": state.current_task}]
    return [
        Send(
            "worker",
            {
                "index": item.get("id", index),
                "task": item.get("task", state.current_task),
                "query": state.current_task,
                "turn": state.turn,
            },
        )
        for index, item in enumerate(plan)
    ]


def _max_concurrency() -> int:
    """How many workers may call the model at once."""
    return max(1, min(settings.max_parallel_agents, MAX_SUBTASKS))


async def run_team(  # noqa: PLR0913, PLR0917 - mirrors run_workflow's options
    query: str,
    thread_id: str = "default",
    llm: Any = None,
    max_subtasks: int = DEFAULT_MAX_SUBTASKS,
    checkpointer: Any = None,
    fresh: bool = False,
) -> AgentState:
    """Plan, fan out to parallel workers, and synthesize one answer."""
    config = {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": _max_concurrency(),
    }

    async with checkpointer_scope(checkpointer) as saver:
        if fresh:
            await reset_thread(saver, thread_id)
        app = create_team_workflow(llm, max_subtasks=max_subtasks, checkpointer=saver)
        if not fresh:
            prior = await resume_info(app, config)
            if prior:
                print(f"Resuming thread {thread_id!r} with {prior} earlier messages")
        result = await app.ainvoke(_initial_state(query, thread_id), config=config)

    return result if isinstance(result, AgentState) else AgentState(**result)


async def run_team_stream(  # noqa: PLR0913, PLR0917 - mirrors run_team's options
    query: str,
    thread_id: str = "default",
    llm: Any = None,
    max_subtasks: int = DEFAULT_MAX_SUBTASKS,
    checkpointer: Any = None,
    fresh: bool = False,
) -> AsyncIterator[AgentState]:
    """Stream team mode, yielding the accumulated state after every step."""
    config = {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": _max_concurrency(),
    }

    async with checkpointer_scope(checkpointer) as saver:
        if fresh:
            await reset_thread(saver, thread_id)
        app = create_team_workflow(llm, max_subtasks=max_subtasks, checkpointer=saver)

        async for event in app.astream(
            _initial_state(query, thread_id), config=config, stream_mode="values"
        ):
            yield event if isinstance(event, AgentState) else AgentState(**event)
