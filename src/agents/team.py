"""Planner, worker and synthesizer agents that make up team mode."""

import json
import re
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.agents import AgentConfig, BaseAgent
from src.graph.state import AgentRole, AgentState

DEFAULT_MAX_SUBTASKS = 4
MAX_SUBTASKS = 20
SYNTHESIS_CHARS_PER_WORKER = 2500
"""How much of each worker's output the synthesizer reads.

Six verbose workers produced ~25k characters and the final merge call stalled on
the gateway. Trimming keeps that call small and cheap without losing the point.
"""

PLANNER_PROMPT = """You are the orchestrator of a team of specialist agents working on one task.

Break the task into independent subtasks that can be worked on at the same time.
Each subtask must be self-contained, concrete, and answerable by one agent on its own.

Reply with JSON only, an array of objects:
[{"task": "what this agent must produce", "tools": ["web_search", "fetch_url"]}]

Rules:
- Between 1 and __MAX_SUBTASKS__ subtasks.
- No overlapping work; each subtask covers a distinct part of the whole.
- Leave "tools" empty when the subtask needs no outside information.
- No prose before or after the JSON."""

WORKER_PROMPT = """You are one specialist agent on a team. You have been given a single subtask.

Rules:
- Answer only your subtask; other agents cover the rest.
- Use your tools when the answer needs current or verifiable outside information.
- Be concrete and specific. State facts, numbers, names and sources.
- If you cannot establish something, say so plainly instead of guessing.
- Keep the result tight: a few short paragraphs or a short list."""

SYNTHESIZER_PROMPT = """You merge the work of a team of specialist agents into one answer.

Rules:
- Lead with the answer to the original task. Not with a description of the process.
- Default to a short reply: at most 150 words, prose or a few bullets. No
  "Executive Summary", no "Conclusion", no headings, no restating the question.
- Write more only when the user explicitly asked for a document, a page, or a
  long-form deliverable.
- Integrate the subtask results; do not concatenate them.
- If they disagree, say which is better supported in one sentence.
- Keep any sources the specialists cited, as short links.
- Never paste HTML or code into the answer. If the task is to build a page, say
  in one line what you built and stop; the page itself goes in the build box."""


def parse_plan(text: str, fallback: str, limit: int) -> list[dict[str, Any]]:
    """Parse the orchestrator's JSON plan, falling back to a single task."""
    match = re.search(r"\[.*\]", text, re.S)
    items: list[Any] = []
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                items = parsed
        except json.JSONDecodeError:
            items = []

    plan: list[dict[str, Any]] = []
    for item in items:
        task: str
        tools: list[Any]
        if isinstance(item, str):
            task, tools = item, []
        elif isinstance(item, dict):
            task = item.get("task") or item.get("title") or ""
            tools = item.get("tools") or []
        else:
            continue
        task = str(task).strip()
        if task:
            plan.append(
                {
                    "id": len(plan),
                    "task": task,
                    "tools": [str(t) for t in tools] if isinstance(tools, list) else [],
                }
            )
        if len(plan) >= limit:
            break

    return plan or [{"id": 0, "task": fallback, "tools": []}]


class PlannerAgent(BaseAgent):
    """Splits the task into parallel subtasks."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        max_subtasks: int = DEFAULT_MAX_SUBTASKS,
    ) -> None:
        config = AgentConfig(
            role=AgentRole.ORCHESTRATOR,
            name="Planner",
            description="Decomposes a task into parallel subtasks",
            system_prompt=PLANNER_PROMPT.replace("__MAX_SUBTASKS__", str(max_subtasks)),
        )
        super().__init__(config, llm)
        self.max_subtasks = max_subtasks

    async def execute(self, state: AgentState) -> dict[str, Any]:
        start = time.time()
        response = await self._ainvoke(
            [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": f"Task: {state.current_task}"},
            ]
        )
        plan = parse_plan(str(response.content), state.current_task, self.max_subtasks)

        return {
            "plan": plan,
            "agent_timings": {"planner": time.time() - start},
            "messages": [self._message(f"Planned {len(plan)} subtasks")],
        }


class WorkerAgent(BaseAgent):
    """Executes one subtask, using tools when the plan asked for them."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        tools: list[BaseTool] | None = None,
    ) -> None:
        from src.config import settings as app_settings

        prompt = WORKER_PROMPT
        if app_settings.computer_use:
            prompt += (
                " You also control this computer: screen_describe shows the desktop "
                "as text with coordinates, screen_shot captures it, and mouse_click, "
                "type_text, press_key and scroll act on it. You cannot see images, so "
                "call screen_describe first, act in small steps, and re-check after "
                "each action."
            )

        config = AgentConfig(
            role=AgentRole.RESEARCHER,
            name="Worker",
            description="Executes one subtask of the plan",
            system_prompt=prompt,
            tools=tools or [],
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        return await self.run_subtask(
            index=0, task=state.current_task, query=state.current_task, turn=state.turn
        )

    async def run_subtask(
        self, index: int, task: str, query: str, turn: str = ""
    ) -> dict[str, Any]:
        """Run one subtask and report it as a state update."""
        start = time.time()
        messages: list[Any] = [
            {"role": "system", "content": self.config.system_prompt},
            {
                "role": "user",
                "content": f"Overall task: {query}. Your subtask: {task}",
            },
        ]

        actor = (f"worker.{index}", f"Worker {index + 1}")
        self._report("planning", actor)

        error: dict[str, Any] | None = None
        try:
            content = await self._with_tools(messages, actor)
        except Exception as e:
            # One worker failing must not throw away the rest of the team's work.
            content = ""
            error = {"agent": f"worker.{index}", "error": str(e), "details": {"task": task}}
            self._report("failed", actor)

        content = content or "(worker produced no output)"
        self._report("done" if error is None else "failed", actor)

        update: dict[str, Any] = {
            "subtask_results": [
                {
                    "index": index,
                    "task": task,
                    "content": content,
                    "duration": time.time() - start,
                    "turn": turn,
                }
            ],
            "agent_timings": {f"worker.{index}": time.time() - start},
            "messages": [self._message(f"Subtask {index + 1} complete: {task[:60]}")],
        }
        if error is not None:
            update["errors"] = [error]
        return update


class SynthesizerAgent(BaseAgent):
    """Merges the subtask results into the final answer."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        config = AgentConfig(
            role=AgentRole.WRITER,
            name="Synthesizer",
            description="Merges parallel worker results into one answer",
            system_prompt=SYNTHESIZER_PROMPT,
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        start = time.time()

        results = "\n\n".join(
            f"[Subtask {item['index'] + 1}] {item['task']}\n{item['content']}"
            for item in sorted(state.subtask_results, key=lambda r: r.get("index", 0))
        )

        response = await self._ainvoke(
            [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Original task: {state.current_task}\n\n"
                        f"Specialist results:\n{results or '(no results returned)'}"
                    ),
                },
            ]
        )

        return {
            "final_report": str(response.content),
            "agent_timings": {"synthesizer": time.time() - start},
            "messages": [self._message("Final answer synthesized")],
        }
