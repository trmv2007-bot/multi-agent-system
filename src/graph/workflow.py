"""LangGraph workflow definition with parallel execution and checkpointing."""

import pathlib
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import PROJECT_ROOT, settings
from src.graph.state import AgentRole, AgentState

if TYPE_CHECKING:  # agents imports this package, so keep it out of the runtime graph
    from src.agents import BaseAgent

try:
    from langgraph.checkpoint.postgres import AsyncPostgresSaver
except ImportError:  # optional dependency: pip install langgraph-checkpoint-postgres
    AsyncPostgresSaver = None  # type: ignore[assignment,misc]

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:  # optional dependency: pip install langgraph-checkpoint-sqlite
    AsyncSqliteSaver = None  # type: ignore[assignment,misc]


RECORD_KEYS = ("research_findings", "analysis_results", "validation_results")
"""The lists an agent appends a turn's work to."""


def create_agent_node(agent: "BaseAgent") -> Any:
    """Wrap one pipeline agent as a graph node scoped to the current turn.

    A resumed thread still holds every earlier turn's findings and analysis, so
    the agent is shown only this turn's records. What it writes is tagged with
    this turn, which is what lets the next node scope its own read the same way.
    """

    async def agent_node(state: AgentState) -> dict[str, Any]:
        mine = state.model_copy(
            update={
                "research_findings": [
                    finding
                    for finding in state.research_findings
                    if finding.get("turn") == state.turn
                ],
                "analysis_results": [
                    analysis
                    for analysis in state.analysis_results
                    if analysis.get("turn") == state.turn
                ],
            }
        )
        update: dict[str, Any] = await agent.execute(mine)
        for key in RECORD_KEYS:
            if key in update:
                update[key] = [{**record, "turn": state.turn} for record in update[key]]
        return update

    return agent_node


def should_continue_research(state: AgentState) -> str:
    """Decide whether to continue research or move to analysis."""
    if not state.research_findings:
        return "researcher"
    return "analyst"


def should_continue_analysis(state: AgentState) -> str:
    """Decide whether to continue analysis or move to validation."""
    if not state.analysis_results:
        return "analyst"
    return "validator"


def should_continue_validation(state: AgentState) -> str:
    """Write the answer, or refuse to when every finding was contradicted.

    This is where the verification grade stops being a report and becomes a
    decision. If the sources contradicted every claim the research step
    produced, there is nothing honest for the writer to write, so the run ends
    on the validator's finding instead.
    """
    from src.agents.verify import CONTRADICTED, verdicts_of

    verdicts = verdicts_of(state.validation_results, state.turn, scope="findings")
    contradicted = [verdict for verdict in verdicts.values() if verdict == CONTRADICTED]
    if verdicts and len(contradicted) == len(verdicts):
        return "abstain"
    return "writer"


def create_abstain_node() -> Any:
    """Close the run honestly when the evidence contradicted itself."""

    async def abstain_node(state: AgentState) -> dict[str, Any]:
        from src.agents.verify import CONTRADICTED, verdicts_of

        verdicts = verdicts_of(state.validation_results, state.turn, scope="findings")
        claims = [claim for claim, verdict in verdicts.items() if verdict == CONTRADICTED]
        listed = "\n".join(f"- {claim[:160]}" for claim in claims[:5])
        report = (
            "I could not verify the evidence for this question, so I am not answering it. "
            f"All {len(claims)} claim(s) the research step returned were contradicted by "
            f"the sources they came from:\n{listed}\n\n"
            "Re-running the research against different sources is the honest next step."
        )
        return {
            "final_report": report,
            "context": {"verification": "abstained", "contradicted_claims": len(claims)},
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Declined to answer: {len(claims)} contradicted claim(s)",
                    "agent": AgentRole.VALIDATOR.value,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ],
        }

    return abstain_node


def is_complete(state: AgentState) -> bool:
    """Check if workflow is complete."""
    return bool(state.final_report)


@asynccontextmanager
async def checkpointer_scope(checkpointer: Any = None) -> AsyncIterator[Any]:
    """
    Yield the checkpointer to compile the graph with.

    ``DATABASE_URL`` selects the durable Postgres saver, which has to stay
    open for the whole run. Without it the run uses the in-memory saver and
    nothing survives the process.
    """
    if checkpointer is not None:
        yield checkpointer
        return

    if settings.database_url is None and settings.checkpoint_db and AsyncSqliteSaver is not None:
        # Durable local state: runs survive a crash and can be resumed. A relative
        # path lives beside the project, not wherever the command was run from.
        db_path = pathlib.Path(settings.checkpoint_db)
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
            await saver.setup()
            yield saver
        return

    if settings.database_url is None:
        yield MemorySaver()
        return

    if AsyncPostgresSaver is None:
        raise RuntimeError(
            "DATABASE_URL is set but langgraph-checkpoint-postgres is not installed. "
            'Install it with: pip install "langgraph-checkpoint-postgres"'
        )

    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
        await saver.setup()
        yield saver


def create_workflow(llm: Any = None, checkpointer: Any = None) -> Any:
    """
    Create the LangGraph workflow.

    Architecture:
    - Researcher runs first
    - Analyst and findings validation run in parallel over the research
    - Analysis validation fans in once both have produced output
    - The grade decides the next step: writer, or abstain when the sources
      contradicted every finding
    - Nodes return partial updates; the state reducers merge them
    """
    from src.agents import ValidatorAgent, create_agents
    from src.tools import as_langchain_tools

    agents = create_agents(llm, as_langchain_tools())
    findings_validator = ValidatorAgent(llm, scope="findings")
    analysis_validator = ValidatorAgent(llm, scope="analysis")

    workflow = StateGraph(AgentState)

    workflow.add_node("researcher", create_agent_node(agents[AgentRole.RESEARCHER]))
    workflow.add_node("analyst", create_agent_node(agents[AgentRole.ANALYST]))
    workflow.add_node("validate_findings", create_agent_node(findings_validator))
    workflow.add_node("validate_analysis", create_agent_node(analysis_validator))
    workflow.add_node("writer", create_agent_node(agents[AgentRole.WRITER]))
    # The gate is a plain function node: it reports what validation found, it
    # does not run an agent.
    workflow.add_node("abstain", create_abstain_node())

    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("researcher", "validate_findings")
    # Both writers of this superstep converge here, so the gate below is
    # evaluated once rather than once per parallel branch.
    workflow.add_edge("analyst", "validate_analysis")
    workflow.add_edge("validate_findings", "validate_analysis")
    workflow.add_conditional_edges(
        "validate_analysis", should_continue_validation, ["writer", "abstain"]
    )
    workflow.add_edge("writer", END)
    workflow.add_edge("abstain", END)

    return workflow.compile(checkpointer=checkpointer or MemorySaver())


def _initial_state(query: str, thread_id: str) -> AgentState:
    """Build the state a run starts from."""
    state = AgentState(thread_id=thread_id, current_task=query, turn=uuid.uuid4().hex)
    state.add_message("user", query)
    return state


async def reset_thread(saver: Any, thread_id: str) -> None:
    """Delete a thread's saved history so the next run starts clean."""
    delete = getattr(saver, "adelete_thread", None)
    if delete is not None:
        await delete(thread_id)


async def resume_info(app: Any, config: dict[str, Any]) -> int:
    """How many messages this thread already holds (0 for a new thread)."""
    try:
        snapshot = await app.aget_state(config)
    except Exception:
        return 0
    values = getattr(snapshot, "values", None) or {}
    messages = values.get("messages") or []
    return len(messages)


async def run_workflow(
    query: str,
    thread_id: str = "default",
    llm: Any = None,
    checkpointer: Any = None,
    fresh: bool = False,
) -> AgentState:
    """Run the complete workflow and return the final state."""
    config = {"configurable": {"thread_id": thread_id}}

    async with checkpointer_scope(checkpointer) as saver:
        if fresh:
            await reset_thread(saver, thread_id)
        app = create_workflow(llm, saver)
        if fresh:
            result = await app.ainvoke(_initial_state(query, thread_id), config=config)
        else:
            prior = await resume_info(app, config)
            if prior:
                print(f"Resuming thread {thread_id!r} with {prior} earlier messages")
            result = await app.ainvoke(_initial_state(query, thread_id), config=config)

    return result if isinstance(result, AgentState) else AgentState(**result)


async def run_workflow_stream(
    query: str,
    thread_id: str = "default",
    llm: Any = None,
    checkpointer: Any = None,
    fresh: bool = False,
) -> AsyncIterator[AgentState]:
    """Run the workflow, yielding the accumulated state after every step."""
    async with checkpointer_scope(checkpointer) as saver:
        if fresh:
            await reset_thread(saver, thread_id)
        app = create_workflow(llm, saver)

        async for event in app.astream(
            _initial_state(query, thread_id),
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="values",
        ):
            yield event if isinstance(event, AgentState) else AgentState(**event)


def get_workflow_graph() -> Any:
    """Get the workflow graph for visualization."""
    return create_workflow().get_graph()
