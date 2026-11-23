"""Base agent, configuration, and the retry/failover policy."""

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.activity import ToolReport, activity
from src.config import get_llm, router, settings
from src.graph.state import AgentRole, AgentState

VERIFIED_CONFIDENCE = 0.7
"""Minimum analysis confidence for a claim to count as verified."""

TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
TOOL_CALL_FUNCTION = re.compile(r"<function=([\w.\-]+)>(.*?)</function>", re.S)
TOOL_CALL_PARAM = re.compile(r"<parameter=([\w_]+)>(.*?)</parameter>", re.S)

HEARTBEAT_SECONDS = 4.0
"""How often a waiting model call re-announces that it is still going."""

RETRYABLE_ERROR_MARKERS = (
    "ratelimit",
    "rate limit",
    "timeout",
    "connection",
    "unavailable",
    "overloaded",
    "internal server error",
)


def _is_retryable(error: Exception) -> bool:
    """Whether a failed model call is worth retrying or failing over from."""
    status = getattr(error, "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    haystack = f"{type(error).__name__} {error}".lower()
    return any(marker in haystack for marker in RETRYABLE_ERROR_MARKERS)


def _is_provider_error(error: Exception) -> bool:
    """Whether the failure came from a provider rather than from our own code."""
    if getattr(error, "status_code", None) is not None:
        return True
    name = type(error).__name__.lower()
    return any(
        marker in name for marker in ("openai", "anthropic", "apierror", "httperror", "google")
    )


def parse_text_tool_calls(text: str) -> list[dict[str, Any]]:
    """
    Read tool calls a model wrote as text instead of returning them structured.

    Some gateway models emit <tool_call><function=name><parameter=x>v</parameter>
    </function></tool_call> in their message body. Without this, the call is lost
    and the markup ends up in the answer.
    """
    calls: list[dict[str, Any]] = []
    for block in TOOL_CALL_BLOCK.findall(text):
        for match in TOOL_CALL_FUNCTION.finditer(block):
            name, body = match.group(1), match.group(2)
            arguments = {key.strip(): value.strip() for key, value in TOOL_CALL_PARAM.findall(body)}
            if not arguments:
                free = body.strip()
                if free:
                    arguments = {"query": free}
            calls.append({"name": name, "arguments": arguments})
    return calls


def strip_tool_markup(text: str) -> str:
    """Remove tool-call markup from text meant for a reader."""
    cleaned = TOOL_CALL_BLOCK.sub(" ", text)
    cleaned = re.sub(r"</?(?:tool_call|function|parameter)[^>]*>", " ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    kept = [line for line in lines if line and not line.startswith(("</", "<parameter"))]
    return "\n".join(kept).strip()


def short_answer_request() -> str:
    """The prompt used when a reply came back empty."""
    return (
        "Your last reply was empty. Answer again now, briefly: at most 60 words, "
        "plain text, no code and no HTML. If the task was to build something, say "
        "what you built and that it is in the build box."
    )


def _payload_error(raw: Any, text: str) -> str:
    """A tool that reports failure in its payload rather than raising."""
    if isinstance(raw, dict):
        return str(raw["error"]) if raw.get("error") else ""

    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return ""
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return ""
    if isinstance(data, dict) and data.get("error"):
        return str(data["error"])
    return ""


def _short_reason(error: Exception) -> str:
    """A few words for the overlay: what went wrong on this attempt."""
    status = getattr(error, "status_code", None)
    if status is not None:
        return f"HTTP {status}"
    return type(error).__name__


def _tool_status(name: str, arguments: dict[str, Any]) -> str:
    """A short "tool: what it is working on" line for the overlay."""
    hint = arguments.get("query") or arguments.get("url") or arguments.get("task") or ""
    hint = str(hint).strip()
    return f"{name}: {hint[:26]}" if hint else name


def _describe(error: Exception) -> str:
    """Surface the provider's own explanation instead of a bare status code."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("message")
        if detail:
            return str(detail)
    return str(error)


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    role: AgentRole
    name: str
    description: str
    system_prompt: str
    tools: list[BaseTool] = Field(default_factory=list)
    max_tokens: int = 4000
    temperature: float = 0.1


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    max_tool_rounds = 2

    def __init__(self, config: AgentConfig, llm: BaseChatModel | None = None):
        self.config = config
        self.llm = llm or get_llm()
        self.tools = config.tools
        self._tool_map = {tool.name: tool for tool in self.tools}

    @abstractmethod
    async def execute(self, state: AgentState) -> dict[str, Any]:
        """
        Execute the agent's task and return a partial state update.

        Nodes must never mutate the state they receive: the reducers in
        AgentState merge each returned key into the graph's channels, so an
        in-place append would be written back a second time.
        """
        raise NotImplementedError

    def _actor(self) -> tuple[str, str]:
        """The cursor key and label this agent reports under."""
        return self.config.role.value, self.config.name

    def _report(self, status: str, actor: tuple[str, str] | None = None) -> None:
        """Show this agent's current step on the overlay and the dashboard."""
        key, label = actor or self._actor()
        activity.update(key, label, status)

    async def _heartbeat(
        self, actor: tuple[str, str] | None, started: float, stop: asyncio.Event
    ) -> None:
        """Count elapsed seconds while a model call is in flight.

        A silent dashboard looks identical to a hung one, so the run must keep
        saying it is still working.
        """
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=HEARTBEAT_SECONDS)
                return
            except TimeoutError:
                self._report(f"thinking · {time.time() - started:.0f}s", actor)

    @property
    def _has_tools(self) -> bool:
        """Whether this agent has any tools bound."""
        return bool(self.tools)

    @staticmethod
    def _text_of(message: Any) -> str:
        """Pull plain text out of a model response, whatever shape it arrives in."""
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block if isinstance(block, str) else str(block.get("text", "")) for block in content
            )
        return str(content)

    async def _with_tools(self, messages: list[Any], actor: tuple[str, str] | None = None) -> str:
        """Call the model, running any tools it asks for."""
        text, _runs = await self._with_tools_traced(messages, actor)
        return text

    async def _with_tools_traced(
        self, messages: list[Any], actor: tuple[str, str] | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Call the model, running any tools it asks for and feeding back the output.

        Returns the written answer plus every tool run, so the caller can
        attribute what the agent actually saw. Nothing is stored on the agent:
        parallel workers share one instance and must not overwrite each other.
        """
        runs: list[dict[str, Any]] = []
        response = await self._ainvoke(messages, with_tools=self._has_tools, actor=actor)

        for _ in range(self.max_tool_rounds):
            tool_calls = list(getattr(response, "tool_calls", None) or [])
            written = self._text_of(response)

            if tool_calls:
                messages.append(response)
            else:
                # Fall back to calls the model wrote into the message body.
                tool_calls = [
                    {"name": call["name"], "args": call["arguments"], "id": None}
                    for call in parse_text_tool_calls(written)
                ]
                if not tool_calls:
                    break
                messages.append({"role": "assistant", "content": written})

            for call in tool_calls:
                arguments = call.get("args") or {}
                output = await self._run_tool(call["name"], arguments, actor)
                runs.append({"name": call["name"], "arguments": arguments, "output": output})
                messages.append(
                    ToolMessage(content=output, tool_call_id=call.get("id") or call["name"])
                )
            response = await self._ainvoke(messages, with_tools=True, actor=actor)

        text = strip_tool_markup(self._text_of(response))
        if not text.strip() and self._has_tools:
            # The last round was tool calls only; ask for the written answer.
            text = strip_tool_markup(self._text_of(await self._ainvoke(messages, actor=actor)))
        if not text.strip():
            # Still nothing: a long answer can be cut off by the token budget.
            # Ask again for something short rather than reporting an empty reply.
            retry = [*messages, {"role": "user", "content": short_answer_request()}]
            text = strip_tool_markup(self._text_of(await self._ainvoke(retry, actor=actor)))
        return text, runs

    async def _run_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        actor: tuple[str, str] | None = None,
    ) -> str:
        """Execute one tool call, returning its output as text."""
        tool = self._tool_map.get(name)
        if tool is None:
            return f"Unknown tool: {name}"

        key, label = actor or self._actor()
        self._report(_tool_status(name, arguments), actor)
        started = time.time()
        attempts = max(1, settings.tool_retries)
        result = ""
        error = ""

        for attempt in range(attempts):
            try:
                raw = await tool.ainvoke(arguments)
                result = str(raw)[:4000]
                error = _payload_error(raw, result)
            except Exception as e:
                error = str(e)
                result = ""

            if not error:
                break
            if attempt == attempts - 1:
                break

            delay = settings.llm_retry_backoff * (2**attempt)
            self._report(
                f"{name} failed · attempt {attempt + 1}/{attempts}, retry in {delay:.0f}s", actor
            )
            await asyncio.sleep(delay)

        if error:
            self._report(f"{name} failed", actor)
            activity.tool(
                ToolReport(
                    key=key,
                    label=label,
                    name=name,
                    arguments=arguments,
                    ok=False,
                    preview=error,
                    seconds=time.time() - started,
                )
            )
            return f"Tool {name} failed: {error}"

        activity.tool(
            ToolReport(
                key=key,
                label=label,
                name=name,
                arguments=arguments,
                ok=True,
                preview=" ".join(result.split())[:400],
                seconds=time.time() - started,
            )
        )
        return result

    def _client(self, with_tools: bool = False) -> Any:
        """The model client to call, optionally bound to the agent's tools."""
        return self.llm.bind_tools(self.tools) if with_tools and self.tools else self.llm

    async def _ainvoke(
        self,
        messages: list[dict[str, Any]],
        with_tools: bool = False,
        actor: tuple[str, str] | None = None,
    ) -> Any:
        """
        Call the model, retrying with exponential backoff and failing over.

        The gateway pools upstream keys, so a 429 usually means "all keys are
        cooling down right now" rather than a permanent refusal: waiting and
        switching providers is the correct response.
        """
        stop = asyncio.Event()
        started = time.time()
        heartbeat = asyncio.create_task(self._heartbeat(actor, started, stop))

        llm = self._client(with_tools)
        chain = router.provider_names() or ["default"]
        attempts = max(3, settings.max_fallback_retries)
        last_error: Exception | None = None
        failures: list[str] = []

        self._report("thinking", actor)
        for attempt in range(attempts):
            try:
                response = await llm.ainvoke(messages)
                stop.set()
                heartbeat.cancel()
                return response
            except Exception as e:
                last_error = e
                failures.append(f"{router.current_provider_name()}: {_short_reason(e)}")
                if not _is_retryable(e) and not _is_provider_error(e):
                    # A bug in our own code, not a provider refusing us.
                    raise
                if attempt == attempts - 1:
                    break

                # Walk the chain one provider per failure; the first full pass
                # costs no sleep, later passes back off exponentially.
                cycle = attempt // len(chain)
                if cycle:
                    delay = settings.llm_retry_backoff * (2 ** (cycle - 1))
                    self._report(
                        f"{_short_reason(e)} · attempt {attempt + 1}/{attempts}, "
                        f"retry in {delay:.0f}s",
                        actor,
                    )
                    await asyncio.sleep(delay)
                else:
                    self._report(
                        f"{_short_reason(e)} · attempt {attempt + 1}/{attempts}, next provider",
                        actor,
                    )

                self.llm = router.next_provider()
                llm = self._client(with_tools)

        assert last_error is not None
        stop.set()
        heartbeat.cancel()
        tried = ", ".join(dict.fromkeys(failures))
        raise RuntimeError(
            f"{self.config.name} could not reach a model after {attempts} attempts "
            f"({tried}): {_describe(last_error)}"
        ) from last_error

    def _message(self, content: str) -> dict[str, Any]:
        """Build an assistant message attributed to this agent."""
        return {
            "role": "assistant",
            "content": content,
            "agent": self.config.role.value,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _build_messages(
        self, state: AgentState, additional_context: str = ""
    ) -> list[dict[str, Any]]:
        """Build messages for LLM call."""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
        ]

        # Add relevant context from state
        if state.research_findings:
            findings_str = "\n".join(
                [
                    f"- {f.get('source', 'unknown')}: {f.get('content', '')[:500]}"
                    for f in state.research_findings[-10:]
                ]
            )
            messages.append({"role": "system", "content": f"RESEARCH FINDINGS:\n{findings_str}"})

        if state.analysis_results:
            analysis_str = "\n".join(
                [
                    f"- {a.get('agent', 'unknown')}: {a.get('content', '')[:500]}"
                    for a in state.analysis_results[-5:]
                ]
            )
            messages.append({"role": "system", "content": f"PREVIOUS ANALYSIS:\n{analysis_str}"})

        if additional_context:
            messages.append(
                {"role": "system", "content": f"ADDITIONAL CONTEXT:\n{additional_context}"}
            )

        # Add conversation history
        messages.extend(state.messages[-20:])  # Last 20 messages

        return messages
