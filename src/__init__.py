"""Multi-Agent System Package."""

__version__ = "0.1.0"
__author__ = "Rishi"

from src.agents import (
    AnalystAgent,
    OrchestratorAgent,
    ResearcherAgent,
    ValidatorAgent,
    WriterAgent,
    create_agents,
)
from src.config import get_llm, settings
from src.graph.state import AgentRole, AgentState
from src.graph.team import run_team
from src.graph.workflow import create_workflow, run_workflow
from src.observability import setup_observability, trace_agent, trace_tool

__all__ = [
    "AgentRole",
    "AgentState",
    "AnalystAgent",
    "OrchestratorAgent",
    "ResearcherAgent",
    "ValidatorAgent",
    "WriterAgent",
    "create_agents",
    "create_workflow",
    "get_llm",
    "run_team",
    "run_workflow",
    "settings",
    "setup_observability",
    "trace_agent",
    "trace_tool",
]
