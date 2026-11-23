"""Tests for the multi-agent system."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents import AgentConfig, AnalystAgent, ResearcherAgent, ValidatorAgent, WriterAgent
from src.config import Settings
from src.graph.state import AgentRole, AgentState


class TestAgentState:
    """Tests for AgentState."""

    def test_initial_state(self):
        state = AgentState(thread_id="test-123")
        assert state.thread_id == "test-123"
        assert state.messages == []
        assert state.research_findings == []
        assert state.final_report == ""

    def test_add_message(self):
        state = AgentState()
        state.add_message("user", "Hello", "test_agent")
        assert len(state.messages) == 1
        assert state.messages[0]["role"] == "user"
        assert state.messages[0]["content"] == "Hello"
        assert state.messages[0]["agent"] == "test_agent"

    def test_add_finding(self):
        state = AgentState()
        state.add_finding({"source": "web", "content": "Test finding"})
        assert len(state.research_findings) == 1
        assert state.research_findings[0]["source"] == "web"

    def test_record_error(self):
        state = AgentState()
        state.record_error("test_agent", "Test error", {"detail": "info"})
        assert len(state.errors) == 1
        assert state.errors[0]["agent"] == "test_agent"
        assert state.errors[0]["error"] == "Test error"

    def test_record_timing(self):
        state = AgentState()
        state.record_timing("researcher", 1.5)
        assert state.agent_timings["researcher"] == 1.5

    def test_record_tokens(self):
        state = AgentState()
        state.record_tokens("researcher", 100)
        state.record_tokens("researcher", 50)
        assert state.token_usage["researcher"] == 150


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        config = AgentConfig(
            role=AgentRole.RESEARCHER,
            name="Test Researcher",
            description="Test agent",
            system_prompt="Test prompt",
        )
        assert config.role == AgentRole.RESEARCHER
        assert config.tools == []
        assert config.temperature == 0.1


class TestResearcherAgent:
    """Tests for ResearcherAgent."""

    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="Research findings here",
                tool_calls=None,
                usage_metadata={"input_tokens": 100, "output_tokens": 50},
            )
        )
        return llm

    @pytest.fixture
    def researcher(self, mock_llm):
        return ResearcherAgent(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_execute_basic(self, researcher):
        state = AgentState(current_task="Test query")
        state.add_message("user", "Test query")

        result = await researcher.execute(state)

        assert len(result["research_findings"]) > 0
        assert result["agent_timings"].get("researcher") is not None
        researcher.llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_tools(self, researcher):
        # Mock tool
        mock_tool = AsyncMock()
        mock_tool.name = "search"
        mock_tool.ainvoke = AsyncMock(return_value="Tool result")
        researcher.tools = [mock_tool]
        researcher._tool_map = {"search": mock_tool}

        # First response asks for the tool, the second writes the answer.
        tool_call_response = MagicMock()
        tool_call_response.content = ""
        tool_call_response.tool_calls = [
            {"name": "search", "args": {"query": "test"}, "id": "call-1"}
        ]
        tool_call_response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

        answer_response = MagicMock()
        answer_response.content = "Answer built from the tool result"
        answer_response.tool_calls = None
        answer_response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

        researcher.llm.bind_tools = MagicMock(return_value=researcher.llm)

        researcher.llm.bind = MagicMock(return_value=researcher.llm)
        researcher.llm.ainvoke = AsyncMock(side_effect=[tool_call_response, answer_response])

        state = AgentState(current_task="Test query")
        result = await researcher.execute(state)

        mock_tool.ainvoke.assert_called_once()
        sources = [finding["source"] for finding in result["research_findings"]]
        # The tool output is attributed to the tool; the write-up is the synthesis.
        assert sources == ["search", "llm_synthesis"]
        assert result["research_findings"][0]["content"] == "Tool result"


class TestAnalystAgent:
    """Tests for AnalystAgent."""

    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="Analysis insight here",
                usage_metadata={"input_tokens": 100, "output_tokens": 50},
            )
        )
        return llm

    @pytest.fixture
    def analyst(self, mock_llm):
        return AnalystAgent(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_execute_with_findings(self, analyst):
        state = AgentState(current_task="Test query")
        state.add_finding({"source": "web", "content": "Finding 1"})
        state.add_finding({"source": "api", "content": "Finding 2"})

        result = await analyst.execute(state)

        assert len(result["analysis_results"]) > 0
        assert result["analysis_results"][0]["insight"] == "Analysis insight here"
        analyst.llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_no_findings(self, analyst):
        state = AgentState(current_task="Test query")

        result = await analyst.execute(state)

        assert len(result["analysis_results"]) == 1
        assert "No research findings" in result["analysis_results"][0]["insight"]


class TestWriterAgent:
    """Tests for WriterAgent."""

    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="# Test Report\n\nExecutive Summary...\n\nDetails...\n\nConclusion...",
                usage_metadata={"input_tokens": 200, "output_tokens": 100},
            )
        )
        return llm

    @pytest.fixture
    def writer(self, mock_llm):
        return WriterAgent(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_execute(self, writer):
        state = AgentState(current_task="Test query")
        state.add_finding({"source": "web", "content": "Finding 1"})
        state.add_analysis({"insight": "Analysis 1", "confidence": 0.9})

        result = await writer.execute(state)

        assert result["final_report"] != ""
        assert "Test Report" in result["final_report"]
        writer.llm.ainvoke.assert_called_once()


class TestValidatorAgent:
    """Tests for ValidatorAgent."""

    @pytest.fixture
    def validator(self):
        return ValidatorAgent(llm=AsyncMock())

    @pytest.mark.asyncio
    async def test_execute_validates_findings(self, validator):
        state = AgentState()
        state.add_finding({"source": "web", "content": "Verified finding"})
        state.add_finding({"source": "llm_synthesis", "content": "Unsourced claim"})

        result = await validator.execute(state)

        # A model that returns nothing gradeable must not confirm anything.
        assert len(result["validation_results"]) >= 2
        assert all(v["status"] != "verified" for v in result["validation_results"])
        assert all(v["verdict"] == "unsupported" for v in result["validation_results"])


class TestWorkflow:
    """Tests for workflow creation."""

    def test_create_workflow(self):
        from src.graph.workflow import create_workflow

        # Mock LLM
        with patch("src.config.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm

            app = create_workflow(mock_llm)

            assert app is not None
            # Check graph has expected nodes
            graph = app.get_graph()
            node_names = [node for node in graph.nodes]
            assert "researcher" in node_names
            assert "analyst" in node_names
            assert "validate_findings" in node_names
            assert "validate_analysis" in node_names
            assert "writer" in node_names

    def test_workflow_edges(self):
        from src.graph.workflow import create_workflow

        with patch("src.config.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm

            app = create_workflow(mock_llm)
            graph = app.get_graph()

            # Check edges exist
            edges = [(e.source, e.target) for e in graph.edges]
            assert ("researcher", "analyst") in edges
            assert ("researcher", "validate_findings") in edges
            assert ("analyst", "validate_analysis") in edges
            assert ("validate_findings", "validate_analysis") in edges
            assert ("validate_analysis", "writer") in edges
            assert ("writer", "__end__") in edges


class TestConfig:
    """Tests for configuration."""

    def test_settings_load(self):
        settings = Settings()
        assert settings.llm_provider in ["keypool", "openai", "anthropic", "google"]
        assert settings.log_level == "INFO"
        # The base URL comes from .env, so only its shape is asserted here.
        assert settings.keypool_base_url.startswith("http://")
        assert settings.keypool_model == "pooled-model"
        # Credentials and infrastructure come from .env or the environment;
        # nothing secret may have a default baked into the source.
        assert Settings.model_fields["keypool_api_key"].default is None
        assert Settings.model_fields["database_url"].default is None

    def test_every_call_goes_to_the_configured_provider(self, monkeypatch):
        from langchain_openai import ChatOpenAI as CO

        from src.config import FreeLLMRouter, settings

        monkeypatch.setattr(settings, "enable_free_fallback", False)
        monkeypatch.setattr(settings, "llm_provider", "keypool")
        monkeypatch.setattr(settings, "keypool_api_key", "test-key")
        monkeypatch.setattr(settings, "groq_api_key", "other-key")

        router = FreeLLMRouter()

        assert router.provider_names() == ["keypool"]
        assert isinstance(router.get_llm(), CO)

    def test_fallbacks_only_appear_when_enabled(self, monkeypatch):
        from src.config import FreeLLMRouter, settings

        monkeypatch.setattr(settings, "enable_free_fallback", True)
        monkeypatch.setattr(settings, "llm_provider", "keypool")
        monkeypatch.setattr(settings, "keypool_api_key", "test-key")
        monkeypatch.setattr(settings, "groq_api_key", "other-key")

        assert FreeLLMRouter().provider_names()[:2] == ["keypool", "groq"]

    def test_missing_primary_key_is_reported_clearly(self, monkeypatch):
        from src.config import FreeLLMRouter, settings

        monkeypatch.setattr(settings, "enable_free_fallback", False)
        monkeypatch.setattr(settings, "llm_provider", "keypool")
        monkeypatch.setattr(settings, "keypool_api_key", None)

        with pytest.raises(RuntimeError, match="LLM_PROVIDER='keypool' is not configured"):
            FreeLLMRouter().get_llm()


def test_server_modules_import():
    """The MCP and A2A servers must import cleanly (they once did not)."""
    import src.a2a.server
    import src.mcp.server

    assert src.a2a.server.A2AAgentServer is not None
    assert src.mcp.server.MCPToolServer is not None


@pytest.mark.asyncio
async def test_nodes_return_updates_without_mutating_input():
    """Nodes must hand LangGraph a delta, never append to the state in place."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content="Mock response",
            tool_calls=None,
            usage_metadata={"input_tokens": 10, "output_tokens": 5},
        )
    )
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    mock_llm.bind = MagicMock(return_value=mock_llm)

    state = AgentState(current_task="Test query")
    state.add_message("user", "Test query")
    before = (len(state.messages), len(state.research_findings))

    update = await ResearcherAgent(llm=mock_llm).execute(state)

    assert isinstance(update, dict)
    assert (len(state.messages), len(state.research_findings)) == before
    assert len(update["research_findings"]) == 1


# Integration-style test
@pytest.mark.asyncio
async def test_full_workflow_mock():
    """Test full workflow with mocked LLM."""
    from src.graph.state import AgentState
    from src.graph.workflow import run_workflow

    with patch("src.agents.base.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        # Return responses without final_report to avoid concurrent write error
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="Mock response",
                tool_calls=None,
                usage_metadata={"input_tokens": 100, "output_tokens": 50},
            )
        )
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm.bind = MagicMock(return_value=mock_llm)
        mock_get_llm.return_value = mock_llm

        result = await run_workflow("Test query", "test-thread")

        # LangGraph returns state as dict, convert to AgentState for assertion
        if isinstance(result, dict):
            result = AgentState(**result)

        assert isinstance(result, AgentState)
        # The writer agent sets final_report, so check it's been set
        assert result.final_report != ""
        assert result.thread_id == "test-thread"

        # Reducers merge each node's delta exactly once. The agents used to
        # mutate the shared state in place, which doubled every list on the
        # way back through the operator.add reducers.
        assert len(result.research_findings) == 1
        assert len(result.analysis_results) == 1
        # One validation for the research finding and one for the analyst's
        # insight. The stubbed model returns prose, which is not a verdict, so
        # both come back unverified rather than trusted.
        assert [v["status"] for v in result.validation_results] == ["unverified", "unverified"]
        assert [v["evidence"] for v in result.validation_results] == [
            ["llm_synthesis"],
            ["llm_synthesis"],
        ]
        assert [m["role"] for m in result.messages] == [
            "user",
            "assistant",
            "assistant",
            "assistant",
            "assistant",
            "assistant",
        ]
        assert result.errors == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestA2AServer:
    """The A2A server must build its agent card and run tasks over HTTP."""

    @pytest.fixture
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient

        from src.a2a.server import A2AAgentServer

        stub_llm = AsyncMock()
        monkeypatch.setattr("src.agents.base.get_llm", MagicMock(return_value=stub_llm))
        server = A2AAgentServer()
        server.agents[AgentRole.VALIDATOR].llm = AsyncMock()
        return TestClient(server.app)

    def test_server_needs_no_credentials_to_build(self, monkeypatch):
        """Discovery and the HTTP app must not depend on a configured provider."""
        from fastapi.testclient import TestClient

        from src.a2a.server import A2AAgentServer

        def no_provider():
            raise RuntimeError("no provider configured")

        monkeypatch.setattr("src.agents.base.get_llm", no_provider)
        server = A2AAgentServer()

        assert server.app is not None
        assert server.agent_card.name == "Multi-Agent System"

        response = TestClient(server.app).get("/.well-known/agent-card.json")
        assert response.status_code == 200

    def test_agent_card(self, client):
        response = client.get("/.well-known/agent.json")
        assert response.status_code == 200
        card = response.json()
        assert card["name"] == "Multi-Agent System"
        assert [s["id"] for s in card["skills"]] == [
            "research",
            "analyze",
            "write",
            "validate",
            "team",
            "full_workflow",
        ]
        assert card["supportedInterfaces"][0]["protocolBinding"] == "JSONRPC"

    def test_task_lifecycle(self, client):
        send = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "tasks/send",
                "params": {
                    "message": {
                        "role": "ROLE_USER",
                        "parts": [{"text": "check this"}],
                        "metadata": {"skill_id": "validate"},
                    },
                    "id": "task-1",
                    "contextId": "ctx-1",
                },
            },
        ).json()
        assert send["result"]["status"]["state"] == "TASK_STATE_COMPLETED"

        get = client.post(
            "/",
            json={"jsonrpc": "2.0", "id": "2", "method": "tasks/get", "params": {"id": "task-1"}},
        ).json()
        assert get["result"]["id"] == "task-1"

        cancel = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "3",
                "method": "tasks/cancel",
                "params": {"id": "task-1"},
            },
        ).json()
        assert cancel["result"]["status"]["state"] == "TASK_STATE_CANCELED"

    def test_unknown_method(self, client):
        response = client.post("/", json={"jsonrpc": "2.0", "id": "4", "method": "nope"})
        assert response.status_code == 400


class TestMCPServer:
    """The MCP HTTP bridge must list and call tools."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from src.mcp.server import create_fastapi_app

        return TestClient(create_fastapi_app())

    def test_tools_list(self, client):
        body = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).json()
        assert [t["name"] for t in body["result"]["tools"]] == [
            "web_search",
            "fetch_url",
            "search_github",
            "get_current_time",
            "research_report",
            "team_task",
        ]

    def test_tools_call(self, client):
        body = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "get_current_time", "arguments": {}},
            },
        ).json()
        assert json.loads(body["result"]["content"][0]["text"])["timezone"] == "UTC"


def stub_router_for(llm):
    """A provider chain with exactly one provider, so no test reaches the network."""
    stub = MagicMock()
    stub.provider_names.return_value = ["stub"]
    stub.next_provider.return_value = llm
    return stub


class RateLimitError(Exception):
    """Stands in for a provider 429."""

    status_code = 429


class TestRetryAndFailover:
    """A rate-limited call must back off and retry instead of killing the run."""

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("src.agents.base.asyncio.sleep", AsyncMock())

        calls = {"count": 0}

        async def flaky(messages):
            calls["count"] += 1
            if calls["count"] < 3:
                raise RateLimitError("all keys cooling down")
            return MagicMock(content="recovered", tool_calls=None, usage_metadata={})

        llm = AsyncMock()
        llm.ainvoke = flaky
        llm.bind_tools = MagicMock(return_value=llm)
        llm.bind = MagicMock(return_value=llm)

        # Failover must not reach the real provider chain from a unit test.
        monkeypatch.setattr("src.agents.base.router", stub_router_for(llm))

        agent = ResearcherAgent(llm=llm)
        update = await agent.execute(AgentState(current_task="q"))

        assert calls["count"] == 3
        assert update["research_findings"][0]["content"] == "recovered"

    @pytest.mark.asyncio
    async def test_retries_exhausted_reports_provider_detail(self, monkeypatch):
        monkeypatch.setattr("src.agents.base.asyncio.sleep", AsyncMock())

        async def always_limited(messages):
            raise RateLimitError("All provider keys are invalid or cooling down")

        llm = AsyncMock()
        llm.ainvoke = always_limited
        llm.bind_tools = MagicMock(return_value=llm)
        llm.bind = MagicMock(return_value=llm)

        monkeypatch.setattr("src.agents.base.router", stub_router_for(llm))

        agent = ResearcherAgent(llm=llm)
        with pytest.raises(RuntimeError, match="All provider keys are invalid"):
            await agent.execute(AgentState(current_task="q"))

    @pytest.mark.asyncio
    async def test_non_retryable_error_is_raised_immediately(self, monkeypatch):
        sleep = AsyncMock()
        monkeypatch.setattr("src.agents.base.asyncio.sleep", sleep)

        async def bad_request(messages):
            raise ValueError("unknown model")

        llm = AsyncMock()
        llm.ainvoke = bad_request
        llm.bind_tools = MagicMock(return_value=llm)
        llm.bind = MagicMock(return_value=llm)

        monkeypatch.setattr("src.agents.base.router", stub_router_for(llm))

        agent = ResearcherAgent(llm=llm)
        with pytest.raises(ValueError, match="unknown model"):
            await agent.execute(AgentState(current_task="q"))
        sleep.assert_not_called()


class TestPlanParsing:
    """The orchestrator's plan has to survive whatever the model returns."""

    def test_reads_json_array(self):
        from src.agents.team import parse_plan

        plan = parse_plan(
            '[{"task": "one", "tools": ["web_search"]}, {"task": "two"}]',
            fallback="fallback",
            limit=4,
        )
        assert [item["task"] for item in plan] == ["one", "two"]
        assert plan[0]["tools"] == ["web_search"]
        assert plan[1]["tools"] == []

    def test_reads_fenced_json_and_caps_length(self):
        from src.agents.team import parse_plan

        text = '```json\n["a", "b", "c", "d", "e"]\n```'
        plan = parse_plan(text, fallback="fallback", limit=3)
        assert [item["task"] for item in plan] == ["a", "b", "c"]

    def test_falls_back_to_the_task(self):
        from src.agents.team import parse_plan

        plan = parse_plan("I could not decide how to split this.", fallback="do it all", limit=4)
        assert plan == [{"id": 0, "task": "do it all", "tools": []}]


class TestTeamWorkflow:
    """Team mode must fan out to parallel workers and synthesize one answer."""

    @staticmethod
    def _llm(plan, answers):
        """A fake model that plans first, then answers each worker call."""
        queue = [plan, *answers, "final synthesis"]
        overlap = {"now": 0, "max": 0}

        async def ainvoke(messages, **kwargs):
            overlap["now"] += 1
            overlap["max"] = max(overlap["max"], overlap["now"])
            await asyncio.sleep(0.02)
            overlap["now"] -= 1
            content = queue.pop(0) if queue else "done"
            return MagicMock(content=content, tool_calls=None, usage_metadata={})

        llm = AsyncMock()
        llm.ainvoke = ainvoke
        llm.bind_tools = MagicMock(return_value=llm)
        llm.bind = MagicMock(return_value=llm)
        return llm, overlap

    @pytest.mark.asyncio
    async def test_fans_out_concurrently(self):
        from src.graph.team import run_team

        plan = json.dumps([{"task": "one"}, {"task": "two"}, {"task": "three"}])
        llm, overlap = self._llm(plan, ["A", "B", "C"])

        with patch("src.agents.base.get_llm", return_value=llm):
            result = await run_team("do three things", "team-thread", llm)

        assert [item["content"] for item in result.subtask_results] == ["A", "B", "C"]
        assert result.final_report == "final synthesis"
        # The workers' calls overlapped instead of running one after another.
        assert overlap["max"] >= 2
        assert [m.get("agent") for m in result.messages].count("researcher") == 3

    @pytest.mark.asyncio
    async def test_worker_runs_tool_calls(self):
        from langchain_core.tools import StructuredTool

        from src.agents.team import WorkerAgent

        searched: list[str] = []

        async def fake_search(query: str) -> str:
            searched.append(query)
            return "fresh fact"

        tool = StructuredTool.from_function(
            coroutine=fake_search, name="web_search", description="search"
        )

        calls = {"n": 0}

        async def ainvoke(messages, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                response = MagicMock(content="", usage_metadata={})
                response.tool_calls = [{"name": "web_search", "args": {"query": "x"}, "id": "c1"}]
                return response
            return MagicMock(content="answer with fresh fact", tool_calls=None, usage_metadata={})

        llm = AsyncMock()
        llm.ainvoke = ainvoke
        llm.bind_tools = MagicMock(return_value=llm)
        llm.bind = MagicMock(return_value=llm)

        worker = WorkerAgent(llm=llm, tools=[tool])
        update = await worker.run_subtask(index=0, task="look it up", query="q")

        assert searched == ["x"]
        assert "fresh fact" in update["subtask_results"][0]["content"]

    @pytest.mark.asyncio
    async def test_worker_asks_for_prose_when_only_tool_calls_came_back(self):
        from src.agents.team import WorkerAgent

        calls = {"n": 0}

        async def ainvoke(messages, **kwargs):
            calls["n"] += 1
            response = MagicMock(usage_metadata={})
            if calls["n"] <= 2:
                response.content = ""
                response.tool_calls = [{"name": "web_search", "args": {"query": "x"}, "id": "c"}]
            else:
                response.content = "the written answer"
                response.tool_calls = None
            return response

        llm = AsyncMock()
        llm.ainvoke = ainvoke
        llm.bind_tools = MagicMock(return_value=llm)
        llm.bind = MagicMock(return_value=llm)

        from langchain_core.tools import StructuredTool

        async def fake_search(query: str) -> str:
            return "tool output"

        tool = StructuredTool.from_function(
            coroutine=fake_search, name="web_search", description="search"
        )

        worker = WorkerAgent(llm=llm, tools=[tool])
        update = await worker.run_subtask(index=0, task="t", query="q")

        assert update["subtask_results"][0]["content"] == "the written answer"

    @pytest.mark.asyncio
    async def test_empty_worker_output_is_marked(self):
        from src.agents.team import WorkerAgent

        async def ainvoke(messages, **kwargs):
            return MagicMock(content="", tool_calls=None, usage_metadata={})

        llm = AsyncMock()
        llm.ainvoke = ainvoke
        llm.bind_tools = MagicMock(return_value=llm)
        llm.bind = MagicMock(return_value=llm)

        update = await WorkerAgent(llm=llm, tools=[]).run_subtask(index=0, task="t", query="q")
        assert update["subtask_results"][0]["content"] == "(worker produced no output)"


class TestAgentFacingTools:
    """Other agents reach this system through MCP, so both entry points must work."""

    @pytest.mark.asyncio
    async def test_research_report_tool(self, monkeypatch):
        from src.graph.state import AgentState
        from src.mcp.server import MCPToolServer

        async def fake_run_workflow(query, thread_id, llm=None, checkpointer=None):
            return AgentState(
                thread_id=thread_id,
                current_task=query,
                final_report="the sourced answer",
                research_findings=[
                    {"source": "web_search", "content": "fact"},
                    {"source": "llm_synthesis", "content": "summary"},
                ],
                validation_results=[{"status": "verified"}, {"status": "unverified"}],
            )

        monkeypatch.setattr("src.graph.workflow.run_workflow", fake_run_workflow)

        result = await MCPToolServer()._research_report("what is x")

        assert result["answer"] == "the sourced answer"
        assert result["sources"] == ["llm_synthesis", "web_search"]
        assert result["validation"] == ["verified", "unverified"]

    @pytest.mark.asyncio
    async def test_team_task_tool(self, monkeypatch):
        from src.graph.state import AgentState
        from src.mcp.server import MCPToolServer

        async def fake_run_team(query, thread_id, llm=None, max_subtasks=4, checkpointer=None):
            return AgentState(
                thread_id=thread_id,
                current_task=query,
                final_report="the assembled answer",
                plan=[{"id": 0, "task": "part one", "tools": []}],
                subtask_results=[{"index": 0, "task": "part one", "content": "abc"}],
            )

        monkeypatch.setattr("src.graph.team.run_team", fake_run_team)

        result = await MCPToolServer()._team_task("do a thing", max_subtasks=2)

        assert result["answer"] == "the assembled answer"
        assert result["plan"] == ["part one"]
        assert result["workers"] == [{"task": "part one", "chars": 3}]

    @pytest.mark.asyncio
    async def test_tool_specs_are_exposed(self):
        from src.mcp.server import MCPToolServer

        names = [tool.name for tool in MCPToolServer().tool_specs()]
        assert names == [
            "web_search",
            "fetch_url",
            "search_github",
            "get_current_time",
            "research_report",
            "team_task",
        ]


class TestAgentCursors:
    """The overlay must be free when off and place one cursor per agent when on."""

    def test_updates_without_listeners_are_harmless(self):
        from src.activity import activity

        assert activity.enabled is False
        # No overlay, no dashboard: publishing must still be free and safe.
        activity.update("researcher", "Researcher", "thinking")
        assert activity.enabled is False
        assert "researcher" in [a["key"] for a in activity.snapshot()]

    def test_tool_status_line(self):
        from src.agents.base import _tool_status

        assert _tool_status("web_search", {"query": "langgraph"}) == "web_search: langgraph"
        assert _tool_status("fetch_url", {"url": "https://example.com"}) == (
            "fetch_url: https://example.com"
        )
        assert _tool_status("get_current_time", {}) == "get_current_time"

    def test_cursors_get_their_own_slot_and_colour(self):
        import queue as queue_module

        from src.overlay import Overlay

        events = queue_module.Queue()
        events.put(("researcher", "Researcher", "thinking", "update"))
        events.put(("worker.0", "Worker 1", "web_search: x", "update"))

        overlay = Overlay(events)
        overlay._screen_w, overlay._screen_h = 1920, 1080
        overlay._drain()

        cursors = list(overlay._cursors.values())
        assert [c.label for c in cursors] == ["Researcher", "Worker 1"]
        assert cursors[1].target_y > cursors[0].target_y
        assert len({c.color for c in cursors}) == 2

    def test_status_change_moves_the_cursor(self):
        import queue as queue_module

        from src.overlay import Overlay

        events = queue_module.Queue()
        overlay = Overlay(events)
        overlay._screen_w, overlay._screen_h = 1920, 1080
        overlay._events.put(("researcher", "Researcher", "thinking", "update"))
        overlay._drain()
        cursor = overlay._cursors["researcher"]
        first_target = cursor.target_x

        overlay._events.put(("researcher", "Researcher", "web_search: x", "update"))
        overlay._drain()

        assert cursor.status == "web_search: x"
        assert cursor.flash > 0
        assert cursor.target_x == first_target  # same slot, only the label changed


class TestThreadMemory:
    """Runs in the same thread must build on each other instead of starting blank."""

    @staticmethod
    def _llm():
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="memo answer", tool_calls=None, usage_metadata={})
        )
        llm.bind_tools = MagicMock(return_value=llm)
        llm.bind = MagicMock(return_value=llm)
        return llm

    @pytest.mark.asyncio
    async def test_second_run_resumes_the_thread(self, tmp_path, monkeypatch):
        from src.config import settings
        from src.graph.workflow import run_workflow

        monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "runs.sqlite"))
        llm = self._llm()

        with patch("src.agents.base.get_llm", return_value=llm):
            first = await run_workflow("first question", "memory-thread", llm, fresh=True)
            second = await run_workflow("second question", "memory-thread", llm)

        assert len(first.messages) == 6
        assert len(second.messages) == 12
        asked = [m["content"] for m in second.messages if m["role"] == "user"]
        assert asked == ["first question", "second question"]

    @pytest.mark.asyncio
    async def test_fresh_run_drops_history(self, tmp_path, monkeypatch):
        from src.config import settings
        from src.graph.workflow import run_workflow

        monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "runs.sqlite"))
        llm = self._llm()

        with patch("src.agents.base.get_llm", return_value=llm):
            await run_workflow("first question", "fresh-thread", llm, fresh=True)
            again = await run_workflow("second question", "fresh-thread", llm, fresh=True)

        asked = [m["content"] for m in again.messages if m["role"] == "user"]
        assert asked == ["second question"]


class TestTeamResilience:
    """One failing worker must not throw away the rest of the team's work."""

    @pytest.mark.asyncio
    async def test_failing_worker_is_recorded_and_the_run_completes(self):
        from src.graph.team import run_team

        plan = json.dumps([{"task": "good part"}, {"task": "BROKEN part"}])

        async def ainvoke(messages, **kwargs):
            text = " ".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
            if "BROKEN" in text and "Specialist results" not in text:
                raise RuntimeError("worker exploded")
            if messages and isinstance(messages[-1], dict) and "Task:" in str(messages[-1]):
                return MagicMock(content=plan, tool_calls=None, usage_metadata={})
            if "Specialist results" in text:
                return MagicMock(content="merged answer", tool_calls=None, usage_metadata={})
            return MagicMock(content="good part answer", tool_calls=None, usage_metadata={})

        llm = AsyncMock()
        llm.ainvoke = ainvoke
        llm.bind_tools = MagicMock(return_value=llm)
        llm.bind = MagicMock(return_value=llm)

        with (
            patch("src.agents.base.get_llm", return_value=llm),
            patch("src.agents.base.router", stub_router_for(llm)),
        ):
            result = await run_team("do the thing", "resilient-thread", llm)

        assert result.final_report == "merged answer"
        assert len(result.errors) == 1
        assert "worker exploded" in result.errors[0]["error"]
        assert len(result.subtask_results) == 2

    @pytest.mark.asyncio
    async def test_fresh_team_run_starts_clean(self):
        """--fresh must not depend on a thread already existing."""
        from src.graph.team import run_team

        plan = json.dumps([{"task": "only part"}])

        async def ainvoke(messages, **kwargs):
            text = str(messages[-1].get("content", "")) if messages else ""
            if "Task:" in text:
                return MagicMock(content=plan, tool_calls=None, usage_metadata={})
            if "Specialist results" in text:
                return MagicMock(content="fresh answer", tool_calls=None, usage_metadata={})
            return MagicMock(content="part answer", tool_calls=None, usage_metadata={})

        llm = AsyncMock()
        llm.ainvoke = ainvoke
        llm.bind_tools = MagicMock(return_value=llm)
        llm.bind = MagicMock(return_value=llm)

        with (
            patch("src.agents.base.get_llm", return_value=llm),
            patch("src.agents.base.router", stub_router_for(llm)),
        ):
            result = await run_team("do it", "fresh-team-thread", llm, fresh=True)

        assert result.final_report == "fresh answer"


class TestDashboard:
    """The dashboard must serve its page and stream a run's progress."""

    def test_page_and_health(self):
        from fastapi.testclient import TestClient

        from src.dashboard.server import create_app

        client = TestClient(create_app())

        page = client.get("/")
        assert page.status_code == 200
        assert "Multi-agent" in page.text
        assert client.get("/health").json()["status"] == "healthy"

    def test_empty_message_is_rejected(self):
        from fastapi.testclient import TestClient

        from src.dashboard.server import create_app

        client = TestClient(create_app())
        body = client.post("/api/chat", json={"thread_id": "t", "text": "   "}).json()
        assert body["error"] == "say something first"

    @pytest.mark.asyncio
    async def test_drive_reports_agents_plan_and_answer(self, monkeypatch):
        from src.dashboard.server import Run, _drive

        plan = json.dumps([{"task": "first part"}, {"task": "second part"}])
        queue = [plan, "part one answer", "part two answer", "the merged answer"]

        async def ainvoke(messages, **kwargs):
            content = queue.pop(0) if queue else "done"
            return MagicMock(content=content, tool_calls=None, usage_metadata={})

        llm = AsyncMock()
        llm.ainvoke = ainvoke
        llm.bind_tools = MagicMock(return_value=llm)
        llm.bind = MagicMock(return_value=llm)
        monkeypatch.setattr("src.config.get_llm", lambda: llm)

        run = Run(id="test", thread_id="chat-test", query="do the thing", mode="team", subtasks=2)
        await _drive(run)

        kinds = [event["type"] for event in run.events]
        assert "agent" in kinds  # the bus reached the browser
        assert "state" in kinds  # the workflow steps did too
        assert kinds[-1] == "done"
        assert run.done is True
        assert run.answer == "the merged answer"
        assert run.plan == ["first part", "second part"]
        labels = {event["label"] for event in run.events if event["type"] == "agent"}
        assert "Planner" in labels


class TestBuildMode:
    """Build mode must turn sections into one page, and keep it valid mid-run."""

    def test_assembly_wraps_fragments(self):
        from src.artifacts import assemble_page

        page = assemble_page(
            "Coffee shop",
            [
                '<section class="hero"><h1>Fresh roast</h1></section>',
                "```html\n<section><h2>Why us</h2></section>\n```",
            ],
        )

        assert page.startswith("<!doctype html>")
        assert page.rstrip().endswith("</html>")
        assert "<h1>Fresh roast</h1>" in page
        assert "<h2>Why us</h2>" in page
        assert "```" not in page

    def test_empty_assembly_is_still_a_page(self):
        from src.artifacts import assemble_page

        page = assemble_page("Nothing yet", [])
        assert page.startswith("<!doctype html>")
        assert "Waiting for the first section" in page

    def test_partial_page_grows_as_sections_land(self):
        from src.artifacts import artifact_event
        from src.graph.state import AgentState

        state = AgentState(current_task="Studio site")
        state.subtask_results = [
            {"index": 0, "task": "hero", "content": '<section class="hero"><h1>One</h1></section>'}
        ]
        first = artifact_event(state, state.current_task)

        state.subtask_results = [
            *state.subtask_results,
            {"index": 1, "task": "features", "content": "<section><h2>Two</h2></section>"},
        ]
        second = artifact_event(state, state.current_task)

        assert "<h1>One</h1>" in first and "<h2>Two</h2>" not in first
        assert "<h1>One</h1>" in second and "<h2>Two</h2>" in second

    def test_plan_parsing_falls_back_to_a_hero(self):
        from src.agents.build import parse_sections

        sections = parse_sections("I cannot plan this.", fallback="coffee shop", limit=5)
        assert sections == [{"id": 0, "section": "hero", "brief": "coffee shop"}]

    @pytest.mark.asyncio
    async def test_build_run_fans_out_and_assembles(self):
        from src.graph.build import run_build

        plan = json.dumps(
            [
                {"section": "hero", "brief": "headline"},
                {"section": "features", "brief": "three cards"},
                {"section": "cta", "brief": "closing"},
            ]
        )
        # plan, research, replan, three sections, review, report
        queue = [
            plan,
            "Coats cost about 40 pounds. SOURCES: https://example.com/coats",
            json.dumps(
                {
                    "feeling": "warm",
                    "palette": {
                        "bg": "#14100c",
                        "surface": "#1e1712",
                        "accent": "#e8a33d",
                        "text": "#f7f0e6",
                        "muted": "#b4a08a",
                    },
                    "radius": "10px",
                }
            ),  # designer brief
            plan,
            '<section class="hero"><h1>Hero</h1></section>',
            "<section><h2>Features</h2></section>",
            '<section class="cta"><h2>Go</h2></section>',
            "[]",
            "Built a three section page.",
        ]

        async def ainvoke(messages, **kwargs):
            return MagicMock(
                content=queue.pop(0) if queue else "", tool_calls=None, usage_metadata={}
            )

        llm = AsyncMock()
        llm.ainvoke = ainvoke
        llm.bind_tools = MagicMock(return_value=llm)
        llm.bind = MagicMock(return_value=llm)

        with (
            patch("src.agents.base.get_llm", return_value=llm),
            patch("src.agents.base.router", stub_router_for(llm)),
        ):
            state = await run_build("A coffee shop page", "build-thread", llm)

        assert len(state.subtask_results) == 3
        assert state.final_report.startswith("<!doctype html>")
        assert all(tag in state.final_report for tag in ("Hero", "Features", ">Go<"))
        # The research pass ran before any section was written.
        assert state.research_findings
        assert "example.com/coats" in str(state.research_findings[0].get("sources"))
        assert "build_summary" in state.context

    @pytest.mark.asyncio
    async def test_review_rewrites_only_the_flagged_section(self):
        from src.graph.build import run_build

        plan = json.dumps(
            [
                {"section": "hero", "brief": "headline"},
                {"section": "cta", "brief": "closing"},
            ]
        )
        review = json.dumps([{"section": "cta", "issue": "the button text is empty"}])
        queue = [
            plan,
            "notes",
            json.dumps(
                {
                    "feeling": "warm",
                    "palette": {
                        "bg": "#14100c",
                        "surface": "#1e1712",
                        "accent": "#e8a33d",
                        "text": "#f7f0e6",
                        "muted": "#b4a08a",
                    },
                    "radius": "10px",
                }
            ),
            plan,
            '<section class="hero"><h1>Hero</h1></section>',
            '<section class="cta"><h2></h2></section>',  # broken
            review,
            '<section class="cta"><h2>Book now</h2></section>',  # fixed
            "[]",  # confirmation round
            "Fixed the button.",
        ]

        async def ainvoke(messages, **kwargs):
            return MagicMock(
                content=queue.pop(0) if queue else "", tool_calls=None, usage_metadata={}
            )

        llm = AsyncMock()
        llm.ainvoke = ainvoke
        llm.bind_tools = MagicMock(return_value=llm)
        llm.bind = MagicMock(return_value=llm)

        with (
            patch("src.agents.base.get_llm", return_value=llm),
            patch("src.agents.base.router", stub_router_for(llm)),
        ):
            state = await run_build("A page", "build-review", llm)

        assert "Book now" in state.final_report
        assert state.validation_results
        rewrites = [r for r in state.subtask_results if r.get("task", "").startswith("cta")]
        assert len(rewrites) == 2


class FlakyTool:
    """A tool that fails a set number of times before answering."""

    def __init__(self, failures, *, mode="raise"):
        self.name = "web_search"
        self.failures = failures
        self.mode = mode
        self.calls = 0

    async def ainvoke(self, arguments):
        self.calls += 1
        if self.calls <= self.failures:
            if self.mode == "raise":
                raise ConnectionError("connection reset")
            return {"error": "upstream 502"}
        return {"results": ["ok"]}


class TestToolRetries:
    """A flaky tool must be retried before the agent gives up on it."""

    @pytest.mark.asyncio
    async def test_retries_a_failing_tool(self, monkeypatch):
        from src.agents import ResearcherAgent

        sleeps = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr("src.agents.base.asyncio.sleep", fake_sleep)

        tool = FlakyTool(failures=1)
        agent = ResearcherAgent(llm=AsyncMock())
        agent._tool_map = {"web_search": tool}

        result = await agent._run_tool("web_search", {"query": "x"})

        assert tool.calls == 2
        assert "ok" in result
        assert sleeps and sleeps[0] > 0

    @pytest.mark.asyncio
    async def test_retries_a_tool_that_reports_an_error(self, monkeypatch):
        from src.agents import ResearcherAgent

        monkeypatch.setattr("src.agents.base.asyncio.sleep", AsyncMock())

        tool = FlakyTool(failures=1, mode="payload")
        agent = ResearcherAgent(llm=AsyncMock())
        agent._tool_map = {"web_search": tool}

        result = await agent._run_tool("web_search", {"query": "x"})

        assert tool.calls == 2
        assert "ok" in result

    @pytest.mark.asyncio
    async def test_gives_up_after_the_budget(self, monkeypatch):
        from src.agents import ResearcherAgent

        monkeypatch.setattr("src.agents.base.asyncio.sleep", AsyncMock())

        tool = FlakyTool(failures=99)
        agent = ResearcherAgent(llm=AsyncMock())
        agent._tool_map = {"web_search": tool}

        result = await agent._run_tool("web_search", {"query": "x"})

        assert tool.calls == 2  # default budget
        assert "failed" in result


class TestTextToolCalls:
    """Models that write tool calls as text must still get their tools run."""

    def test_parses_the_written_form(self):
        from src.agents.base import parse_text_tool_calls

        text = (
            "Let me look.\n"
            "<tool_call><function=fetch_url>"
            "<parameter=url>https://example.com/a</parameter>"
            "<parameter=max_chars>4000</parameter>"
            "</function></tool_call>"
        )
        calls = parse_text_tool_calls(text)

        assert calls == [
            {
                "name": "fetch_url",
                "arguments": {"url": "https://example.com/a", "max_chars": "4000"},
            }
        ]

    def test_strips_markup_from_prose(self):
        from src.agents.base import strip_tool_markup

        text = (
            "Claude has connectors.\n"
            "<tool_call><function=fetch_url><parameter=url>https://x.dev</parameter>"
            "</function></tool_call>\n"
            "It also has MCP."
        )
        cleaned = strip_tool_markup(text)

        assert "<tool_call>" not in cleaned
        assert "<parameter" not in cleaned
        assert "Claude has connectors." in cleaned
        assert "It also has MCP." in cleaned

    @pytest.mark.asyncio
    async def test_written_calls_are_executed(self):
        from langchain_core.tools import StructuredTool

        from src.agents.team import WorkerAgent

        opened = []

        async def fake_fetch(url: str) -> str:
            opened.append(url)
            return "page contents"

        tool = StructuredTool.from_function(
            coroutine=fake_fetch, name="fetch_url", description="fetch"
        )

        calls = {"n": 0}

        async def ainvoke(messages, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return MagicMock(
                    content=(
                        "<tool_call><function=fetch_url>"
                        "<parameter=url>https://example.com</parameter>"
                        "</function></tool_call>"
                    ),
                    tool_calls=None,
                    usage_metadata={},
                )
            return MagicMock(
                content="Final answer with page contents.", tool_calls=None, usage_metadata={}
            )

        llm = AsyncMock()
        llm.ainvoke = ainvoke
        llm.bind_tools = MagicMock(return_value=llm)
        llm.bind = MagicMock(return_value=llm)

        worker = WorkerAgent(llm=llm, tools=[tool])
        update = await worker.run_subtask(index=0, task="t", query="q")

        assert opened == ["https://example.com"]
        text = update["subtask_results"][0]["content"]
        assert "<tool_call>" not in text
        assert "Final answer" in text


class FakePyAutoGUI:
    """Records input calls instead of touching a real desktop."""

    def __init__(self):
        self.calls = []
        self.size_value = type("S", (), {"width": 1920, "height": 1080})()
        self.pos = type("P", (), {"x": 10, "y": 20})()

    def click(self, **kwargs):
        self.calls.append(("click", kwargs))

    def moveTo(self, x, y, duration=0):
        self.calls.append(("move", x, y))

    def press(self, key):
        self.calls.append(("press", key))

    def hotkey(self, *keys):
        self.calls.append(("hotkey", keys))

    def scroll(self, amount):
        self.calls.append(("scroll", amount))

    def typewrite(self, text, interval=0):
        self.calls.append(("typewrite", text))

    def position(self):
        return self.pos

    def size(self):
        return self.size_value

    def screenshot(self):
        self.calls.append(("screenshot",))
        raise RuntimeError("no screen in tests")


class TestComputerUse:
    """Computer use stays off unless asked, and input is spent against a budget."""

    def test_off_by_default(self, monkeypatch):
        from src import computer
        from src.config import settings

        monkeypatch.setattr(settings, "computer_use", False)
        with pytest.raises(computer.ComputerUnavailable, match="Computer use is off"):
            computer.screen_shot()

    def test_click_spends_the_budget(self, monkeypatch):
        from src import computer
        from src.config import settings

        fake = FakePyAutoGUI()
        monkeypatch.setattr(settings, "computer_use", True)
        monkeypatch.setattr(settings, "computer_max_actions", 2)
        monkeypatch.setattr(computer, "_guard", lambda: fake)
        computer.reset_session()

        computer.mouse_click(100, 200)
        computer.press_key("enter")

        with pytest.raises(computer.ComputerUnavailable, match="budget reached"):
            computer.mouse_click(10, 10)

        assert fake.calls[0][0] == "click"
        assert fake.calls[1] == ("press", "enter")
        assert computer.actions_used() == 2

    def test_chord_is_pressed_as_hotkey(self, monkeypatch):
        from src import computer
        from src.config import settings

        fake = FakePyAutoGUI()
        monkeypatch.setattr(settings, "computer_use", True)
        monkeypatch.setattr(computer, "_guard", lambda: fake)

        computer.press_key("ctrl+s")
        assert fake.calls == [("hotkey", ("ctrl", "s"))]

    def test_tools_report_an_error_rather_than_raising(self, monkeypatch):
        import asyncio
        import json

        from src import computer
        from src.config import settings

        monkeypatch.setattr(settings, "computer_use", False)
        click_tool = next(t for t in computer.computer_tools() if t.name == "mouse_click")
        raw = asyncio.run(click_tool.ainvoke({"x": 1, "y": 1}))

        assert "Computer use is off" in json.loads(raw)["error"]

    def test_computer_tools_are_hidden_until_enabled(self, monkeypatch):
        from src.config import settings
        from src.tools import as_langchain_tools

        monkeypatch.setattr(settings, "computer_use", False)
        assert not [t for t in as_langchain_tools() if t.name == "screen_describe"]

        monkeypatch.setattr(settings, "computer_use", True)
        names = [t.name for t in as_langchain_tools()]
        assert "screen_describe" in names
        assert "mouse_click" in names


class TestTranscriptCleaning:
    """Whisper invents noise tags; they must not reach a model or the screen."""

    def test_strips_noise_tags(self):
        from src.voice import clean_transcript

        assert (
            clean_transcript("Set a reminder [BLANK_AUDIO] for tomorrow")
            == "Set a reminder for tomorrow"
        )
        assert clean_transcript("(wind blowing) what is due friday") == "what is due friday"

    def test_silence_becomes_empty(self):
        from src.voice import clean_transcript

        assert clean_transcript("[BLANK_AUDIO]") == ""
        assert clean_transcript("  )  ") == ""

    def test_stop_words_are_recognised(self):
        from src.voice import is_stop_command

        for phrase in ["stop", "Stop.", "ok stop", "never mind", "cancel", "that's all"]:
            assert is_stop_command(phrase), phrase

    def test_questions_are_not_stop_words(self):
        from src.voice import is_stop_command

        for phrase in ["stop the server on port 3000", "what is due friday", "cancel my 9am class"]:
            assert not is_stop_command(phrase), phrase


class TestDesignThemes:
    """Two different subjects must never come out looking identical."""

    def test_parses_a_designer_theme(self):
        from src.design import parse_theme

        theme = parse_theme(
            '```json\n{"feeling": "cold desert night", "palette": {"bg": "#050810", '
            '"accent": "#7fd4ff"}, "radius": "2px", "background_css": "background: #050810;"}\n```'
        )
        assert theme["palette"]["accent"] == "#7fd4ff"
        assert theme["palette"]["bg"] == "#050810"
        assert theme["radius"] == "2px"
        assert "050810" in theme["background_css"]

    def test_rejects_bad_values_and_falls_back(self):
        from src.design import DEFAULT_THEME, parse_theme

        theme = parse_theme('{"palette": {"accent": "blue", "bg": "#123456"}}')
        assert theme["palette"]["accent"] == DEFAULT_THEME["palette"]["accent"]
        assert theme["palette"]["bg"] == "#123456"

    def test_fallback_differs_by_subject(self):
        from src.design import fallback_theme

        themes = {
            fallback_theme(text)["palette"]["accent"]
            for text in (
                "an observatory",
                "a bakery",
                "a bookshop",
                "a gym",
                "a tea shop",
                "a studio",
            )
        }
        assert len(themes) > 1

    def test_theme_reaches_the_page(self):
        from src.artifacts import assemble_page
        from src.design import theme_css

        theme = {
            "palette": {
                "bg": "#050810",
                "surface": "#0b1220",
                "accent": "#7fd4ff",
                "text": "#eaf2ff",
                "muted": "#8fa0bd",
            },
            "radius": "2px",
            "background_css": "background: radial-gradient(#10233d, #050810);",
        }
        page = assemble_page("Observatory", ["<section><h1>Stars</h1></section>"], theme=theme)

        assert "--accent: #7fd4ff" in page
        assert "radial-gradient(#10233d" in page
        assert "--radius: 2px" in page
        assert theme_css(theme) in page
