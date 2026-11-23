# Using this system from other AI agents (MCP)

The server speaks MCP over stdio, so any MCP client can start it as a subprocess
and call the agents like local tools: Claude Desktop, Claude Code, Hermes, or your
own agent loop.

## What it exposes

| Tool | What it does | Typical time |
|---|---|---|
| `research_report` | Runs the research → analysis → fact-check → write pipeline and returns one cited report | 30-90s |
| `team_task` | Plans the task, runs specialist subagents in parallel with tools, synthesizes one answer | 30-90s |
| `web_search` | Live DuckDuckGo results (title, URL, snippet) | 1-3s |
| `fetch_url` | Fetch a page and return its text | 1-5s |
| `search_github` | GitHub repository or issue search | 1-3s |
| `get_current_time` | Current UTC time | instant |

`research_report` returns `{answer, sources, validation, timings}`; `team_task`
returns `{answer, plan, workers, timings}`. The calling agent gets the answer plus
enough of the trace to judge it.

## Requirements

The server needs the project's `.env`, which is where the KeyPool key lives, and
your KeyPool gateway must be running (`python -c "from src.config import check_keypool; print(check_keypool())"`).
`run_mcp.py` changes into the project directory on start, so `.env` loads whichever
directory the client launches it from.

## Claude Desktop

Add this to `%APPDATA%\Claude\claude_desktop_config.json` (create the file if it
does not exist), then restart Claude:

```json
{
  "mcpServers": {
    "multi-agent": {
      "command": "C:\\Users\\rishi\\AppData\\Local\\Programs\\Python\\Python313\\python.exe",
      "args": ["C:\\Users\\rishi\\multi-agent-system\\run_mcp.py"]
    }
  }
}
```

## Claude Code

Add it to the project, which writes `.mcp.json` in the repository root:

```bash
cd C:\Users\rishi\multi-agent-system
claude mcp add multi-agent -- "C:\Users\rishi\AppData\Local\Programs\Python\Python313\python.exe" "C:\Users\rishi\multi-agent-system\run_mcp.py"
```

Or commit this `.mcp.json` so every session in the repo gets it:

```json
{
  "mcpServers": {
    "multi-agent": {
      "type": "stdio",
      "command": "C:\\Users\\rishi\\AppData\\Local\\Programs\\Python\\Python313\\python.exe",
      "args": ["C:\\Users\\rishi\\multi-agent-system\\run_mcp.py"]
    }
  }
}
```

## Any other MCP client

Two options, both tested:

**stdio** - launch `python C:\Users\rishi\multi-agent-system\run_mcp.py` and speak
JSON-RPC over its stdin/stdout. If the client needs a single executable, install
the entry point once with `pip install -e .` and use `multi-agent-mcp` instead.

**HTTP** - start the server as a long-running process:

```bash
python -m src.mcp.server --http
```

That serves JSON-RPC at `POST http://127.0.0.1:8001/mcp` with `tools/list` and
`tools/call`, plus `GET /health`. Useful for agents that cannot spawn subprocesses.

## Checking the connection without an agent

Run the test suite, which starts the server the same way a client does and calls
its tools:

```bash
pytest tests/ -q -k mcp
```

## Cost and latency

Every `research_report` call is five model calls (researcher, analyst, two
validators' checks, writer); every `team_task` call is one planning call, one per
subtask, and one synthesis. A four-subtask team asks your gateway for six
completions. If the gateway is rate-limiting, calls back off and retry before
failing, so a slow answer is normal and an error means the pool was empty for the
whole retry window.
