"""Agent mode: one capable agent that acts, rather than a team that plans.

This is the mode for "do the thing on my computer": a single model loop that can
search, read pages, and — when computer use is on — look at the desktop and drive
the mouse and keyboard, working through as many tool rounds as it needs before it
answers.
"""

import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from src.agents.base import AgentConfig, BaseAgent
from src.config import settings
from src.graph.state import AgentRole, AgentState
from src.graph.workflow import _initial_state, checkpointer_scope, reset_thread, resume_info
from src.tools import as_langchain_tools

AGENT_PROMPT = """You are an agent running on the user's own Windows PC. You get things done.

You can search the web, open pages, search GitHub{computer}.

How to work:
- Take the smallest next step that makes progress, then check the result.
- Prefer tools over memory for anything current or specific to this machine.
- When something fails, read the error and adjust instead of repeating it.
- If the task is to build a web page, write it and hand it over with the
  build_page tool. Never paste markup into your reply.
- Stop when the task is done. Do not keep going to look busy.
- Then answer in at most 120 words: what you did, what the result was, and anything
  the user must know. No headings, no essay."""

COMPUTER_NOTE = (
    ", and you control this computer: screen_describe shows the desktop as text "
    "with coordinates, then mouse_click, type_text, press_key and scroll act on it"
)


class SoloAgent(BaseAgent):
    """One agent that works through a task with tools until it is done."""

    max_tool_rounds = 6

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        prompt = AGENT_PROMPT.replace("computer", COMPUTER_NOTE if settings.computer_use else "")
        config = AgentConfig(
            role=AgentRole.ORCHESTRATOR,
            name="Agent",
            description="Works a task end to end with tools",
            system_prompt=prompt,
            tools=as_langchain_tools(),
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        """Work the task, reporting every tool the agent reaches for."""
        start = time.time()
        actor = ("agent", "Agent")
        self._report("working", actor)

        history = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": state.current_task},
        ]

        error: dict[str, Any] | None = None
        try:
            answer, runs = await self._with_tools_traced(history, actor)
        except Exception as e:
            answer, runs = "", []
            error = {"agent": "agent", "error": str(e), "details": {"task": state.current_task}}
            self._report("failed", actor)

        self._report("done" if error is None else "failed", actor)

        findings = [
            {
                "source": run["name"],
                "content": str(run["output"])[:800],
                "confidence": 0.8,
                "relevance": 0.8,
                "turn": state.turn,
            }
            for run in runs
        ]

        update: dict[str, Any] = {
            "final_report": answer.strip() or "(the agent produced no answer)",
            "research_findings": findings,
            "agent_timings": {"agent": time.time() - start},
            "messages": [self._message(f"Worked the task with {len(runs)} tool calls")],
        }
        if error is not None:
            update["errors"] = [error]
        return update


def create_agent_workflow(llm: Any = None, checkpointer: Any = None) -> Any:
    """Build the single-agent graph."""
    agent = SoloAgent(llm)

    async def agent_node(state: AgentState) -> dict[str, Any]:
        return await agent.execute(state)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=checkpointer)


async def run_agent(
    query: str,
    thread_id: str = "default",
    llm: Any = None,
    checkpointer: Any = None,
    fresh: bool = False,
) -> AgentState:
    """One agent works the task end to end."""
    config = {"configurable": {"thread_id": thread_id}}

    async with checkpointer_scope(checkpointer) as saver:
        if fresh:
            await reset_thread(saver, thread_id)
        app = create_agent_workflow(llm, checkpointer=saver)
        if not fresh:
            prior = await resume_info(app, config)
            if prior:
                print(f"Resuming thread {thread_id!r} with {prior} earlier messages")
        result = await app.ainvoke(_initial_state(query, thread_id), config=config)

    return result if isinstance(result, AgentState) else AgentState(**result)


async def run_agent_stream(
    query: str,
    thread_id: str = "default",
    llm: Any = None,
    checkpointer: Any = None,
    fresh: bool = False,
) -> AsyncIterator[AgentState]:
    """Stream an agent run."""
    config = {"configurable": {"thread_id": thread_id}}

    async with checkpointer_scope(checkpointer) as saver:
        if fresh:
            await reset_thread(saver, thread_id)
        app = create_agent_workflow(llm, checkpointer=saver)
        async for event in app.astream(
            _initial_state(query, thread_id), config=config, stream_mode="values"
        ):
            yield event if isinstance(event, AgentState) else AgentState(**event)
