"""Main entry point for the multi-agent system."""

import argparse
import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from src.activity import activity
from src.agents.team import DEFAULT_MAX_SUBTASKS
from src.config import get_llm, router, settings
from src.graph.build import run_build, run_build_stream
from src.graph.state import AgentState
from src.graph.team import run_team, run_team_stream
from src.graph.workflow import run_workflow, run_workflow_stream
from src.observability import end_run, format_run_summary, start_run

# Which settings field holds the model id for each provider.
MODEL_FIELDS = {
    "keypool": "keypool_model",
    "openrouter": "openrouter_free_model",
    "groq": "groq_free_model",
    "google": "google_free_model",
    "cloudflare": "cloudflare_free_model",
    "openai": "openai_model",
    "local": "local_llm_model",
}


@dataclass(frozen=True)
class RunOptions:
    """How to run the system for one query."""

    stream: bool = False
    team: bool = False
    build: bool = False
    subtasks: int = DEFAULT_MAX_SUBTASKS
    cursor: bool = False
    fresh: bool = False

    @property
    def mode(self) -> str:
        """Which workflow this run uses."""
        if self.build:
            return "build"
        return "team" if self.team else "pipeline"


MODE_LABELS = {
    "build": "build (live page)",
    "team": "team (parallel subagents)",
    "pipeline": "pipeline",
}

STREAM_RUNNERS: dict[str, Callable[..., AsyncIterator[AgentState]]] = {
    "build": lambda query, thread, llm, limit, fresh: run_build_stream(
        query, thread, llm, limit, fresh=fresh
    ),
    "team": lambda query, thread, llm, limit, fresh: run_team_stream(
        query, thread, llm, limit, fresh=fresh
    ),
    "pipeline": lambda query, thread, llm, limit, fresh: run_workflow_stream(
        query, thread, llm, fresh=fresh
    ),
}

CALL_RUNNERS: dict[str, Callable[..., Any]] = {
    "build": lambda query, thread, llm, limit, fresh: run_build(
        query, thread, llm, limit, fresh=fresh
    ),
    "team": lambda query, thread, llm, limit, fresh: run_team(
        query, thread, llm, limit, fresh=fresh
    ),
    "pipeline": lambda query, thread, llm, limit, fresh: run_workflow(
        query, thread, llm, fresh=fresh
    ),
}

PROGRESS: dict[str, Callable[[AgentState], str]] = {
    "build": lambda e: f"  {len(e.subtask_results)} sections built",
    "team": lambda e: (
        f"  {len(e.plan)} planned, {len(e.subtask_results)} done, "
        f"answer {len(e.final_report)} chars"
    ),
    "pipeline": lambda e: (
        f"  {len(e.research_findings)} findings, {len(e.analysis_results)} analysis, "
        f"{len(e.validation_results)} validations, report {len(e.final_report)} chars"
    ),
}


def _utf8_stdout() -> None:
    """Keep Windows consoles from crashing on non-cp1252 output."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


async def _print_providers() -> None:
    """Report which providers in the failover chain are answering right now."""
    print(f"Provider chain: {router.provider_names()}")
    print(f"Primary:        {settings.llm_provider}\n")
    for provider in router.providers:
        alive = await router.check(provider)
        print(f"  {'ok      ' if alive else 'unusable'}  {provider.name:<11} {provider.model}")


def _print_report(result: AgentState) -> None:
    """Print the final answer and what it cost."""
    print(f"\n{'=' * 60}")
    print("FINAL ANSWER")
    print(f"{'=' * 60}")
    print(result.final_report or "No answer generated")
    print(f"{'=' * 60}")

    if result.plan:
        print("\nPLAN")
        for item in result.plan:
            label = item.get("section") or item.get("task", "")
            print(f"  {item.get('id', 0) + 1}. {label}")
    if result.subtask_results:
        print("\nWORKERS")
        for item in sorted(result.subtask_results, key=lambda r: r.get("index", 0)):
            print(f"  {item['index'] + 1}. {len(item['content'])} chars in {item['duration']:.1f}s")

    print("\nSTATISTICS")
    print(f"  Research Findings: {len(result.research_findings)}")
    print(f"  Analysis Results:  {len(result.analysis_results)}")
    print(f"  Validation Checks: {len(result.validation_results)}")
    print(f"  Agent Timings:     {result.agent_timings}")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
        for err in result.errors:
            print(f"    - {err['agent']}: {err['error']}")


async def _run(query: str, thread_id: str, llm: Any, options: RunOptions) -> AgentState:
    """Drive the chosen mode, printing progress when streaming."""
    if not options.stream:
        final = await CALL_RUNNERS[options.mode](
            query, thread_id, llm, options.subtasks, options.fresh
        )
        return final if isinstance(final, AgentState) else AgentState(**final)

    stream = STREAM_RUNNERS[options.mode](query, thread_id, llm, options.subtasks, options.fresh)
    result: AgentState | None = None
    async for event in stream:
        result = event
        print(PROGRESS[options.mode](event))

    if result is None:
        raise RuntimeError("workflow produced no state")
    return result


async def main(
    query: str,
    thread_id: str | None = None,
    options: RunOptions | None = None,
) -> AgentState:
    """Run the multi-agent system with a query."""
    options = options or RunOptions()
    thread_id = thread_id or settings.thread_id
    llm = get_llm()
    start_run(f"{options.mode} · {query[:60]}")

    if options.cursor:
        activity.enable()
        print("Cursors:  on - each agent appears on screen as it works")

    print(f"Mode:     {MODE_LABELS[options.mode]}")
    print(f"Query:    {query}")
    print(f"Thread:   {thread_id}")
    print(f"Provider: {router.current_provider_name()}")
    print()

    result = await _run(query, thread_id, llm, options)

    _print_report(result)
    print(f"  Cost:              {format_run_summary(end_run())}")
    if options.cursor:
        activity.disable()
    return result


def cli() -> None:
    """Command-line interface."""
    _utf8_stdout()

    parser = argparse.ArgumentParser(description="Multi-Agent System")
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--thread", "-t", help="Thread ID for session")
    parser.add_argument("--stream", "-s", action="store_true", help="Print each step as it lands")
    parser.add_argument(
        "--team",
        action="store_true",
        help="Orchestrate parallel subagents instead of running the research pipeline",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build a web page: parallel sections assembled into one HTML file",
    )
    parser.add_argument(
        "--subtasks",
        type=int,
        default=DEFAULT_MAX_SUBTASKS,
        help="Maximum subtasks the orchestrator may plan (or sections to build)",
    )
    parser.add_argument("--workers", type=int, help="Maximum workers running at once")
    parser.add_argument(
        "--providers",
        action="store_true",
        help="Check which providers in the chain answer, then exit",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore anything saved under this thread id and start clean",
    )
    parser.add_argument(
        "--cursor",
        action="store_true",
        help="Show a live on-screen cursor for every agent as it works",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=sorted(MODEL_FIELDS),
        help="LLM provider to route through",
    )
    parser.add_argument("--model", "-m", help="Model name for the chosen provider")

    args = parser.parse_args()

    if args.providers:
        asyncio.run(_print_providers())
        return

    if args.provider:
        settings.llm_provider = args.provider
    if args.model:
        field = MODEL_FIELDS[args.provider or settings.llm_provider]
        setattr(settings, field, args.model)
    if args.workers:
        settings.max_parallel_agents = args.workers
    if args.provider or args.model:
        # The router builds its clients at import time, so rebuild them now.
        router.reset()

    query = args.query
    if not query:
        if sys.stdin.isatty():
            query = input("Enter your query: ").strip()
        else:
            query = sys.stdin.read().strip()

    if not query:
        print("Error: No query provided")
        sys.exit(1)

    try:
        options = RunOptions(
            stream=args.stream,
            team=args.team,
            build=args.build,
            subtasks=args.subtasks,
            cursor=args.cursor,
            fresh=args.fresh,
        )
        asyncio.run(main(query, args.thread, options))
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()
