"""LangGraph state definition with reducers for parallel merging."""

import operator
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field


def _latest(x: Any, y: Any) -> Any:
    """Reducer that keeps the value written by the most recent step."""
    return y


def _merge_dict(x: dict[str, Any], y: dict[str, Any]) -> dict[str, Any]:
    """Reducer that merges dicts, newer keys winning."""
    return {**x, **y}


def _keep_first(x: Any, _y: Any) -> Any:
    """Reducer that keeps the first value written."""
    return x


class AgentRole(str, Enum):
    """Roles for specialized agents."""

    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    VALIDATOR = "validator"
    ORCHESTRATOR = "orchestrator"


class TaskStatus(str, Enum):
    """Status of a task in the workflow."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentState(BaseModel):
    """
    Unified state for the multi-agent system.

    All agents read from and write to this shared state.
    Reducers (via Annotated) automatically merge parallel writes.
    """

    # Conversation history - auto-merges with add_messages reducer
    messages: Annotated[list[dict[str, Any]], operator.add] = Field(
        default_factory=list, description="Full conversation history"
    )

    # Research findings - parallel agents append via operator.add
    research_findings: Annotated[list[dict[str, Any]], operator.add] = Field(
        default_factory=list, description="Findings from research agents"
    )

    # Analysis results - parallel agents append
    analysis_results: Annotated[list[dict[str, Any]], operator.add] = Field(
        default_factory=list, description="Analysis from analyst agents"
    )

    # Validation results
    validation_results: Annotated[list[dict[str, Any]], operator.add] = Field(
        default_factory=list, description="Validation checks results"
    )

    # Team mode: the plan produced by the orchestrator and each worker's result
    plan: Annotated[list[dict[str, Any]], _latest] = Field(
        default_factory=list, description="Subtasks the orchestrator decided to run"
    )
    subtask_results: Annotated[list[dict[str, Any]], operator.add] = Field(
        default_factory=list, description="Results reported by parallel workers"
    )

    # Final synthesized output - use custom reducer to take last value
    final_report: Annotated[str, _latest] = Field(
        default="", description="Final synthesized report"
    )

    # Metadata
    thread_id: Annotated[str, _latest] = Field(
        default="default-session", description="Thread/session identifier"
    )
    current_task: Annotated[str, _latest] = Field(
        default="", description="Current task being executed"
    )
    turn: Annotated[str, _latest] = Field(
        default="",
        description="Id of the current turn, so one turn's results are not reused by the next",
    )
    task_status: Annotated[dict[str, TaskStatus], _merge_dict] = Field(
        default_factory=dict, description="Status of each task"
    )
    agent_timings: Annotated[dict[str, float], _merge_dict] = Field(
        default_factory=dict, description="Execution time per agent"
    )
    token_usage: Annotated[dict[str, int], _merge_dict] = Field(
        default_factory=dict, description="Token usage per agent"
    )
    errors: Annotated[list[dict[str, Any]], operator.add] = Field(
        default_factory=list, description="Errors encountered"
    )

    # Context for handoffs
    context: Annotated[dict[str, Any], _merge_dict] = Field(
        default_factory=dict, description="Shared context for handoffs"
    )

    # Metadata
    created_at: Annotated[datetime, _keep_first] = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: Annotated[datetime, _latest] = Field(default_factory=lambda: datetime.now(UTC))

    def update_timestamp(self) -> None:
        """Update the timestamp."""
        self.updated_at = datetime.now(UTC)

    def add_message(self, role: str, content: str, agent: str | None = None) -> None:
        """Add a message to the conversation."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if agent:
            msg["agent"] = agent
        self.messages.append(msg)
        self.update_timestamp()

    def add_finding(self, finding: dict[str, Any]) -> None:
        """Add a research finding."""
        finding["timestamp"] = datetime.now(UTC).isoformat()
        self.research_findings.append(finding)
        self.update_timestamp()

    def add_analysis(self, analysis: dict[str, Any]) -> None:
        """Add an analysis result."""
        analysis["timestamp"] = datetime.now(UTC).isoformat()
        self.analysis_results.append(analysis)
        self.update_timestamp()

    def add_validation(self, validation: dict[str, Any]) -> None:
        """Add a validation result."""
        validation["timestamp"] = datetime.now(UTC).isoformat()
        self.validation_results.append(validation)
        self.update_timestamp()

    def record_error(self, agent: str, error: str, details: dict | None = None) -> None:
        """Record an error."""
        self.errors.append(
            {
                "agent": agent,
                "error": error,
                "details": details or {},
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        self.update_timestamp()

    def record_timing(self, agent: str, duration: float) -> None:
        """Record agent execution time."""
        self.agent_timings[agent] = duration
        self.update_timestamp()

    def record_tokens(self, agent: str, tokens: int) -> None:
        """Record token usage."""
        self.token_usage[agent] = self.token_usage.get(agent, 0) + tokens
        self.update_timestamp()


# Type alias for the state schema used in LangGraph
StateSchema = AgentState
