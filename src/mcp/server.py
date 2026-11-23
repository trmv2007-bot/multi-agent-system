"""MCP Server implementation for tool sharing.

The tool implementations live in ``src.tools`` so the in-process agents and
this server share one definition; this module only adds the MCP and HTTP
transports on top of them.
"""

import argparse
import asyncio
import json
import uuid
from typing import Any

from fastapi import FastAPI, Request
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from mcp.server import Server
from src import tools
from src.config import get_tracer, settings


class MCPToolServer:
    """MCP Server that exposes tools to agents."""

    def __init__(self) -> None:
        self.server = Server("multi-agent-tools")
        self._handlers = {
            "web_search": lambda arguments: tools.web_search(
                arguments["query"], arguments.get("max_results", 5)
            ),
            "fetch_url": lambda arguments: tools.fetch_url(
                arguments["url"], arguments.get("max_chars", 5000)
            ),
            "gather_sources": lambda arguments: tools.gather_sources(
                arguments["query"], arguments.get("max_sources", 4)
            ),
            "search_github": lambda arguments: tools.search_github(
                arguments["query"], arguments.get("type", "repositories")
            ),
            "get_current_time": lambda _arguments: tools.current_time(),
            "research_report": lambda arguments: self._research_report(
                arguments["query"], arguments.get("thread_id")
            ),
            "team_task": lambda arguments: self._team_task(
                arguments["task"], int(arguments.get("max_subtasks", 4))
            ),
        }
        self._register_tools()

    def tool_specs(self) -> list[Tool]:
        """Declare the tools this server exposes."""
        return [
            Tool(
                name="web_search",
                description="Search the web for current information",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum results",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="fetch_url",
                description="Fetch content from a URL",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                        "max_chars": {
                            "type": "integer",
                            "default": 5000,
                            "description": "Max characters to return",
                        },
                    },
                    "required": ["url"],
                },
            ),
            Tool(
                name="gather_sources",
                description=(
                    "Search, open the top results, and return only the pages that are about "
                    "the query, one per site, documentation first"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_sources": {
                            "type": "integer",
                            "default": 4,
                            "description": "Maximum pages to return",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="search_github",
                description="Search GitHub for repositories, code, or issues",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "type": {
                            "type": "string",
                            "enum": ["repositories", "code", "issues"],
                            "default": "repositories",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_current_time",
                description="Get current date and time",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="research_report",
                description=(
                    "Run the multi-agent research pipeline: research with live web and GitHub "
                    "tools, analyse, fact-check, then write one cited report. Slow (30-90s)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to research"},
                        "thread_id": {
                            "type": "string",
                            "description": "Optional session id to group related runs",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="team_task",
                description=(
                    "Run the multi-agent team: an orchestrator splits the task into subtasks, "
                    "specialist subagents work on them in parallel with tools, and one "
                    "synthesizer merges the results. Slow (30-90s)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "The task to complete"},
                        "max_subtasks": {
                            "type": "integer",
                            "default": 4,
                            "description": "Upper bound on parallel subtasks",
                        },
                    },
                    "required": ["task"],
                },
            ),
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Run one tool by name. Shared by the stdio and HTTP transports."""
        with get_tracer().start_as_current_span(f"mcp.tool.{name}") as span:
            span.set_attribute("tool.name", name)

            try:
                handler = self._handlers[name]
                result = await handler(arguments)
            except KeyError:
                return _error(f"Unknown tool: {name}")
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("success", False)
                return _error(str(e))

            span.set_attribute("success", True)
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(result))])

    async def _research_report(self, query: str, thread_id: str | None = None) -> dict[str, Any]:
        """Run the research pipeline and report the answer with its sources."""
        from src.graph.workflow import run_workflow

        state = await run_workflow(query, thread_id or f"mcp-{uuid.uuid4()}")

        return {
            "answer": state.final_report,
            "sources": sorted({finding.get("source", "") for finding in state.research_findings}),
            "validation": [check.get("status", "") for check in state.validation_results],
            "timings": state.agent_timings,
        }

    async def _team_task(self, task: str, max_subtasks: int = 4) -> dict[str, Any]:
        """Run team mode and report the answer alongside the plan it worked from."""
        from src.graph.team import run_team

        state = await run_team(task, f"mcp-{uuid.uuid4()}", max_subtasks=max_subtasks)

        return {
            "answer": state.final_report,
            "plan": [item.get("task", "") for item in state.plan],
            "workers": [
                {"task": item.get("task", ""), "chars": len(item.get("content", ""))}
                for item in sorted(state.subtask_results, key=lambda r: r.get("index", 0))
            ],
            "timings": state.agent_timings,
        }

    def _register_tools(self) -> None:
        """Wire the stdio protocol handlers to the tool implementations."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return self.tool_specs()

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
            return await self.call_tool(name, arguments)

    async def run(self) -> None:
        """Run the MCP server over stdio."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )


def _error(message: str) -> CallToolResult:
    """Build an error result."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps({"error": message}))],
        isError=True,
    )


# FastAPI wrapper for HTTP transport
def create_fastapi_app() -> FastAPI:
    """Create FastAPI app for HTTP-based MCP server."""
    app = FastAPI(title="MCP Tool Server")
    mcp_server = MCPToolServer()

    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> dict[str, Any]:
        body = await request.json()
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "result": {"tools": [t.model_dump() for t in mcp_server.tool_specs()]},
                "id": request_id,
            }

        if method == "tools/call":
            result = await mcp_server.call_tool(params["name"], params.get("arguments", {}))
            return {"jsonrpc": "2.0", "result": result.model_dump(), "id": request_id}

        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": "Method not found"},
            "id": request_id,
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    return app


async def run_mcp_server(http: bool = False) -> None:
    """Run the MCP server over stdio, or over HTTP when http is set."""
    if not http:
        await MCPToolServer().run()
        return

    import uvicorn

    config = uvicorn.Config(
        create_fastapi_app(),
        host=settings.mcp_server_host,
        port=settings.mcp_server_port,
    )
    await uvicorn.Server(config).serve()


def main() -> None:
    """Entry point for MCP clients that launch this as a subprocess."""
    parser = argparse.ArgumentParser(description="Multi-agent system MCP server")
    parser.add_argument("--http", action="store_true", help="Serve over HTTP instead of stdio")
    args = parser.parse_args()
    asyncio.run(run_mcp_server(http=args.http))


if __name__ == "__main__":
    main()
