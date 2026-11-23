"""The MCP and A2A HTTP servers stay on this machine, and need a token to leave it.

Localhost with no token has to keep behaving exactly as before, so every
protected case here is paired with an unprotected one.
"""

import sys

import pytest
from fastapi.testclient import TestClient

from src.a2a.server import A2AAgentServer, run_a2a_server
from src.config import Settings, settings
from src.mcp.server import create_fastapi_app, run_mcp_server
from src.server_guard import is_loopback, require_authenticated_bind

TOKEN = "s3cret-token"


@pytest.fixture
def mcp_client_factory():
    """Build an MCP HTTP app with an explicit token, so tests never depend on .env."""

    def build(token: str | None) -> TestClient:
        return TestClient(create_fastapi_app(auth_token=token))

    return build


def test_both_servers_default_to_loopback():
    assert Settings.model_fields["mcp_server_host"].default == "127.0.0.1"
    assert Settings.model_fields["a2a_server_host"].default == "127.0.0.1"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2"])
def test_loopback_hosts_are_recognised(host):
    assert is_loopback(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com"])
def test_network_hosts_are_not_loopback(host):
    assert not is_loopback(host)


def test_loopback_starts_without_a_token():
    require_authenticated_bind("127.0.0.1", None, server="MCP", host_env="MCP_SERVER_HOST")


def test_network_bind_without_a_token_refuses_to_start():
    with pytest.raises(SystemExit) as refused:
        require_authenticated_bind("0.0.0.0", None, server="MCP", host_env="MCP_SERVER_HOST")

    message = str(refused.value)
    assert "Refusing to start the MCP server" in message
    assert "SERVER_AUTH_TOKEN" in message
    assert "MCP_SERVER_HOST" in message


def test_network_bind_with_a_token_starts():
    require_authenticated_bind("0.0.0.0", TOKEN, server="MCP", host_env="MCP_SERVER_HOST")


def test_mcp_localhost_without_a_token_is_unchanged(mcp_client_factory):
    client = mcp_client_factory(None)

    assert client.get("/health").json() == {"status": "healthy"}

    response = client.post("/mcp", json={"id": 1, "method": "tools/list"})
    assert response.status_code == 200
    assert response.json()["result"]["tools"]


def test_mcp_request_without_the_token_is_rejected(mcp_client_factory):
    client = mcp_client_factory(TOKEN)

    response = client.post("/mcp", json={"id": 1, "method": "tools/list"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert client.get("/health").status_code == 401


def test_mcp_request_with_a_wrong_token_is_rejected(mcp_client_factory):
    client = mcp_client_factory(TOKEN)

    response = client.post(
        "/mcp",
        json={"id": 1, "method": "tools/list"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401


def test_mcp_request_with_the_token_succeeds(mcp_client_factory):
    client = mcp_client_factory(TOKEN)

    response = client.post(
        "/mcp",
        json={"id": 1, "method": "tools/list"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 200
    assert [tool["name"] for tool in response.json()["result"]["tools"]]


def test_a2a_localhost_without_a_token_is_unchanged():
    client = TestClient(A2AAgentServer().app)

    assert client.get("/health").status_code == 200
    assert client.get("/.well-known/agent.json").json()["name"] == "Multi-Agent System"


def test_a2a_request_without_the_token_is_rejected():
    client = TestClient(A2AAgentServer(auth_token=TOKEN).app)

    response = client.get("/.well-known/agent.json")
    assert response.status_code == 401

    rejected = client.post("/", json={"jsonrpc": "2.0", "method": "tasks/get"})
    assert rejected.status_code == 401


def test_a2a_request_with_the_token_succeeds():
    client = TestClient(A2AAgentServer(auth_token=TOKEN).app)

    response = client.get("/.well-known/agent.json", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200
    assert response.json()["name"] == "Multi-Agent System"


async def test_mcp_http_startup_refuses_a_network_bind_without_a_token(monkeypatch):
    monkeypatch.setattr(settings, "mcp_server_host", "0.0.0.0")
    monkeypatch.setattr(settings, "server_auth_token", None)

    with pytest.raises(SystemExit):
        await run_mcp_server(http=True)


async def test_a2a_http_startup_refuses_a_network_bind_without_a_token(monkeypatch):
    monkeypatch.setattr(settings, "a2a_server_host", "0.0.0.0")
    monkeypatch.setattr(settings, "server_auth_token", None)
    monkeypatch.setattr(sys, "argv", ["src.a2a.server", "--http"])

    with pytest.raises(SystemExit):
        await run_a2a_server()
