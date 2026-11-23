"""A2A (Agent-to-Agent) Protocol server and client."""

import json
import uuid
from typing import Any

import httpx
import uvicorn
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel, Field

from src.agents import AgentRole, BaseAgent, create_agents
from src.config import get_tracer, settings
from src.graph.state import AgentState

A2A_PROTOCOL_VERSION = "1.0"

# A2A skill ids are the public verb names; AgentRole is the internal enum.
SKILL_TO_ROLE = {
    "research": AgentRole.RESEARCHER,
    "analyze": AgentRole.ANALYST,
    "write": AgentRole.WRITER,
    "validate": AgentRole.VALIDATOR,
}

# Which key each skill reports under, and where its results live in the state.
SKILL_RESULT_KEYS = {
    "research": "findings",
    "analyze": "analysis",
    "validate": "validations",
}
SKILL_STATE_KEYS = {
    "research": "research_findings",
    "analyze": "analysis_results",
    "validate": "validation_results",
}


def _to_json(message: Any) -> dict[str, Any]:
    """Serialize a protobuf A2A message to the spec's camelCase JSON."""
    return MessageToDict(message, preserving_proto_field_name=False)


class A2ARequest(BaseModel):
    """A2A JSON-RPC request."""

    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None


class A2AResponse(BaseModel):
    """A2A JSON-RPC response."""

    jsonrpc: str = "2.0"
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    id: str | None = None


class A2AAgentServer:
    """A2A Server that exposes agents as remote services."""

    def __init__(self) -> None:
        self._agents: dict[AgentRole, BaseAgent] | None = None
        self.skills = self._define_skills()
        self.agent_card = self._create_agent_card()
        self.tasks: dict[str, Task] = {}
        self.app = self._create_app()

    @property
    def agents(self) -> dict[AgentRole, BaseAgent]:
        """The agent instances, built on first use.

        Construction reaches for the provider chain, so it must not run while a
        server is being created: an agent card, the HTTP app, and discovery all
        work with no credentials configured.
        """
        if self._agents is None:
            self._agents = create_agents()
        return self._agents

    def _define_skills(self) -> list[AgentSkill]:
        """Define the skills this agent server provides."""
        return [
            AgentSkill(
                id="research",
                name="Research",
                description="Gather information from multiple sources",
                tags=["research", "search"],
                input_modes=["text"],
                output_modes=["data"],
            ),
            AgentSkill(
                id="analyze",
                name="Analyze",
                description="Analyze findings and extract insights",
                tags=["analysis"],
                input_modes=["data"],
                output_modes=["data"],
            ),
            AgentSkill(
                id="write",
                name="Write",
                description="Generate reports from analysis",
                tags=["writing"],
                input_modes=["data"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="validate",
                name="Validate",
                description="Validate findings and check for hallucinations",
                tags=["validation"],
                input_modes=["data"],
                output_modes=["data"],
            ),
            AgentSkill(
                id="team",
                name="Team",
                description="Plan the task, run specialist subagents in parallel, synthesize one answer",
                tags=["team", "parallel"],
                input_modes=["text"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="full_workflow",
                name="Full Workflow",
                description="Execute complete research-analyze-write-validate workflow",
                tags=["workflow"],
                input_modes=["text"],
                output_modes=["text"],
            ),
        ]

    def _create_agent_card(self) -> AgentCard:
        """Create the agent card for discovery."""
        return AgentCard(
            name="Multi-Agent System",
            description=(
                "A production-ready multi-agent system with research, analysis, "
                "writing, and validation capabilities"
            ),
            version="1.0.0",
            supported_interfaces=[
                AgentInterface(
                    url=f"http://{settings.a2a_server_host}:{settings.a2a_server_port}",
                    protocol_binding="JSONRPC",
                    protocol_version=A2A_PROTOCOL_VERSION,
                )
            ],
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=False),
            skills=self.skills,
        )

    def _create_app(self) -> FastAPI:
        """Create FastAPI app with A2A endpoints."""
        app = FastAPI(title="A2A Agent Server")

        @app.get("/.well-known/agent.json")
        async def agent_card() -> dict[str, Any]:
            return _to_json(self.agent_card)

        @app.get("/.well-known/agent-card.json")
        async def agent_card_alt() -> dict[str, Any]:
            return _to_json(self.agent_card)

        @app.post("/")
        async def handle_jsonrpc(request: Request) -> JSONResponse:
            body = await request.json()
            return await self._handle_request(body)

        @app.get("/health")
        async def health() -> dict[str, Any]:
            return {"status": "healthy", "agent": self.agent_card.name}

        return app

    async def _handle_request(self, body: dict[str, Any]) -> JSONResponse:
        """Handle A2A JSON-RPC request."""
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")

        with get_tracer().start_as_current_span(f"a2a.{method}") as span:
            span.set_attribute("a2a.method", method or "unknown")
            span.set_attribute("a2a.request_id", request_id or "unknown")

            try:
                if method in ("tasks/send", "tasks/sendSubscribe"):
                    result = await self._handle_task_send(params)
                elif method == "tasks/get":
                    result = self._handle_task_get(params)
                elif method == "tasks/cancel":
                    result = self._handle_task_cancel(params)
                else:
                    return JSONResponse(
                        A2AResponse(
                            error={"code": -32601, "message": f"Unknown method: {method}"},
                            id=request_id,
                        ).model_dump(),
                        status_code=400,
                    )

                return JSONResponse(A2AResponse(result=result, id=request_id).model_dump())

            except Exception as e:
                span.record_exception(e)
                return JSONResponse(
                    A2AResponse(
                        error={"code": -32603, "message": str(e)}, id=request_id
                    ).model_dump(),
                    status_code=500,
                )

    @staticmethod
    def _text_of(message: dict[str, Any]) -> str:
        """Concatenate the text parts of an incoming message."""
        return "".join(part.get("text", "") for part in message.get("parts", []) if "text" in part)

    async def _handle_task_send(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tasks/send request."""
        message = params.get("message", {})
        task_id = params.get("id") or str(uuid.uuid4())
        context_id = params.get("contextId") or str(uuid.uuid4())
        text = self._text_of(message)
        skill_id = (message.get("metadata") or {}).get("skill_id", "full_workflow")

        task = Task(id=task_id, context_id=context_id, status=TaskStatus())
        task.status.state = TaskState.TASK_STATE_WORKING
        self.tasks[task_id] = task

        try:
            result = await self._execute_skill(skill_id, text, context_id)
            task.artifacts.append(
                Artifact(artifact_id=str(uuid.uuid4()), parts=[Part(text=result)])
            )
            task.status.state = TaskState.TASK_STATE_COMPLETED
            return _to_json(task)

        except Exception as e:
            task.status.state = TaskState.TASK_STATE_FAILED
            task.status.message.CopyFrom(
                Message(
                    message_id=str(uuid.uuid4()),
                    role=Role.ROLE_AGENT,
                    parts=[Part(text=str(e))],
                )
            )
            return _to_json(task)

    def _handle_task_get(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tasks/get request."""
        task = self.tasks.get(params.get("id") or "")
        if task is None:
            raise ValueError(f"Task not found: {params.get('id')}")
        return _to_json(task)

    def _handle_task_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tasks/cancel request."""
        task = self.tasks.get(params.get("id") or "")
        if task is None:
            raise ValueError(f"Task not found: {params.get('id')}")
        task.status.state = TaskState.TASK_STATE_CANCELED
        return _to_json(task)

    async def _execute_skill(self, skill_id: str, query: str, context_id: str) -> str:
        """Execute a specific skill."""
        from src.graph.workflow import run_workflow

        if skill_id == "full_workflow":
            result = await run_workflow(query, context_id)
            return result.final_report or "Workflow completed but no report generated"

        if skill_id == "team":
            from src.graph.team import run_team

            team = await run_team(query, context_id)
            answer = team.final_report or "Team completed but produced no answer"
            if team.errors:
                return f"{answer} ({len(team.errors)} subtask(s) failed: {team.errors[0]['error']})"
            return answer

        role = SKILL_TO_ROLE.get(skill_id)
        if role is None:
            raise ValueError(f"Unknown skill: {skill_id}")

        state = AgentState(thread_id=context_id, current_task=query)
        state.add_message("user", query)
        update = await self.agents[role].execute(state)

        if skill_id == "write":
            return str(update.get("final_report", ""))
        key = SKILL_RESULT_KEYS[skill_id]
        return json.dumps({key: update.get(SKILL_STATE_KEYS[skill_id], [])})


class A2AAgentClient:
    """A2A Client for calling remote agents."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.agent_card: dict[str, Any] | None = None
        self._skills_cache: dict[str, dict[str, Any]] = {}

    async def discover(self) -> dict[str, Any]:
        """Discover the remote agent card."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/.well-known/agent.json")
            response.raise_for_status()
            self.agent_card = response.json()
            for skill in self.agent_card.get("skills", []):
                self._skills_cache[skill["id"]] = skill
            return self.agent_card

    async def send_task(
        self,
        skill_id: str,
        query: str,
        context_id: str | None = None,
    ) -> str:
        """Send a task to the remote agent and return its text result."""
        if not self.agent_card:
            await self.discover()

        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": query}],
                    "metadata": {"skill_id": skill_id},
                },
                "id": str(uuid.uuid4()),
                "contextId": context_id or str(uuid.uuid4()),
            },
            "id": str(uuid.uuid4()),
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self.base_url}/", json=request)
            response.raise_for_status()
            result = response.json()

        if "error" in result:
            raise RuntimeError(f"A2A error: {result['error']}")

        task_result = result.get("result", {})
        artifacts = task_result.get("artifacts", [])
        if artifacts:
            parts = artifacts[0].get("parts", [])
            if parts:
                return str(parts[0].get("text", ""))

        status_message = task_result.get("status", {}).get("message", {})
        parts = status_message.get("parts", [])
        return str(parts[0].get("text", "No result")) if parts else "No result"

    def get_skill(self, skill_id: str) -> dict[str, Any] | None:
        """Get a skill by ID."""
        return self._skills_cache.get(skill_id)

    def list_skills(self) -> list[dict[str, Any]]:
        """List all available skills."""
        return list(self._skills_cache.values())


async def run_a2a_server() -> None:
    """Run the A2A server."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        server = A2AAgentServer()
        config = uvicorn.Config(
            server.app,
            host=settings.a2a_server_host,
            port=settings.a2a_server_port,
        )
        await uvicorn.Server(config).serve()
    else:
        print("Use --http flag to run as HTTP server")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_a2a_server())
