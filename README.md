# Multi-Agent System: LangGraph + MCP + A2A + OTel

A production-ready multi-agent system that feels like a single agent but operates as a coordinated team of specialized agents.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED MULTI-AGENT SYSTEM                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                    LANGGRAPH STATE GRAPH                        │   │
│   │  TypedState + Reducers = SINGLE SHARED MEMORY                  │   │
│   │  Checkpointer (Postgres) = Durability, Resume, Time-Travel     │   │
│   │  Send API = Parallel Fan-out with Automatic Merge              │   │
│   └────────────────────────────────────────────────────────────────┘   │
│         │                    ▲                    ▲                    │
│         ▼                    │                    │                    │
│   ┌───────────┐        ┌───────────┐        ┌───────────┐            │
│   │ RESEARCHER│        │  ANALYST  │        │  WRITER   │            │
│   │  Agent    │        │  Agent    │        │  Agent    │            │
│   └─────┬─────┘        └─────┬─────┘        └─────┬─────┘            │
│         │                    │                    │                    │
│         ▼                    ▼                    ▼                    │
│   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐     │
│   │ MCP SERVER       │ │ A2A REMOTE       │ │ VALIDATOR        │     │
│   │ (Web Search,     │ │ (External Agent) │ │ Agent            │     │
│   │  GitHub, URLs)   │ │                  │ │                  │     │
│   └──────────────────┘ └──────────────────┘ └──────────────────┘     │
│                                                                         │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │              OTEL / LANGSMITH OBSERVABILITY                     │   │
│   │  Full span hierarchy: app → session → agent → trace → span    │   │
│   └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

- **🧠 Unified State**: Single `TypedDict` state with reducers for automatic parallel merge
- **💾 Durability**: every step is checkpointed to SQLite (or Postgres), so runs survive a crash and a thread remembers what you told it earlier
- **⚡ Parallel Execution**: Research → Analyst + findings validation (parallel) → analysis validation → Writer, or an honest abstain when every finding was contradicted
- **👥 Team Mode**: an orchestrator splits a task into subtasks and LangGraph's `Send` API fans them out to concurrent workers, which use the same tools as the MCP server, before one synthesizer merges the results
- **🔧 MCP Tools**: Standardized tool access (web search, GitHub, URL fetch) via MCP
- **🤝 A2A Protocol**: Expose agents as remote services, call external agents
- **📊 Observability**: OpenTelemetry + LangSmith tracing with full span hierarchy
- **✅ Verification Gates**: SAST, type checking, linting and tests gate a change; every run's claims are graded before the writer sees them
- **🛡️ Hallucination Detection**: a model call per validation pass grades each finding against the text its source actually returned. Contradicted claims never reach the writer, unverified ones are labelled in the answer, and a run whose findings were all contradicted declines to answer. Cost: one extra small completion per pass (two in pipeline mode)

## Quick Start

```bash
pip install -e ".[dev]"

# Configure: .env is read from the project root, wherever you run from
cp .env.example .env

# Check the model chain before spending a run
multi-agent --providers

# Run
multi-agent "Analyze the current state of AI agents in 2026"
```

`pip install -e .` also puts `multi-agent` and `multi-agent-mcp` on your PATH, so
neither the CLI nor the MCP server depends on the working directory.

## Local Mode (Ollama, LM Studio, LocalAI)

To use a local LLM instead of KeyPool or cloud APIs:

```bash
# Edit .env to use local provider
LLM_PROVIDER=local
LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
LOCAL_LLM_MODEL=llama3.2

# Run
make run QUERY="Analyze AI agents"
```

**Ollama setup:**
```bash
# Start Ollama
ollama serve

# Pull a model
ollama pull llama3.2

# Verify it's running
curl http://localhost:11434/v1/models
```

**LM Studio setup:**
1. Start LM Studio local server (default port 1234)
2. Set `LOCAL_LLM_BASE_URL=http://127.0.0.1:1234/v1`
3. Set `LLM_PROVIDER=local`

**LocalAI setup:**
1. Start LocalAI server
2. Set `LOCAL_LLM_BASE_URL=http://localhost:8080/v1`
3. Set `LLM_PROVIDER=local`

## Check Provider Connectivity

```python
from src.config import check_local_llm, check_keypool, get_router

if check_local_llm():
    print("Local LLM is reachable")

if check_keypool():
    print("KeyPool gateway is reachable")

# See all providers in the failover chain
router = get_router()
print("Provider chain:", router.provider_names())
print("Current:", router.current_provider_name())
```

## Provider Routing — KeyPool Only, Fallbacks Opt-In

Every model call goes through `LLM_PROVIDER` and nothing else. With the default
`LLM_PROVIDER=keypool`, that is KeyPool for every call, and a rate limit is
handled by retrying KeyPool with exponential backoff rather than by quietly
switching to a different backend.

Set `ENABLE_FREE_FALLBACK=true` to add the other providers you hold keys for as
fallbacks. The chain then becomes:

1. **KeyPool** (your gateway) — primary
2. **OpenRouter** `:free` models
3. **Groq** free tier
4. **Google Gemini** free tier
5. **Cloudflare Workers AI**
6. **OpenAI**, then **Local Ollama** as the last resort

Run `multi-agent --providers` to see which links answer. Only providers with a
configured key appear at all.

**To add free providers**, edit `.env` and add API keys:

```bash
# KeyPool (primary - already configured)
LLM_PROVIDER=keypool
KEYPOOL_BASE_URL=http://127.0.0.1:8787
KEYPOOL_API_KEY=kp_your_keypool_key_here
KEYPOOL_MODEL=pooled-model

# Add free fallbacks (optional - add keys to expand the chain)
OPENROUTER_API_KEY=sk-or-v1-...  # https://openrouter.ai/keys
GROQ_API_KEY=gsk_...             # https://console.groq.com/keys
GOOGLE_API_KEY=AI...             # https://aistudio.google.com/app/apikey
CLOUDFLARE_API_KEY=...           # https://dash.cloudflare.com/profile/api-tokens
CLOUDFLARE_ACCOUNT_ID=...

# Local fallback (always in chain as last resort)
LLM_PROVIDER=local
LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
LOCAL_LLM_MODEL=llama3.2
```

**Usage:**
```python
from src.config import get_llm, get_router, router

# Primary provider (KeyPool by default)
llm = get_llm()

# Specific provider
llm = get_llm("groq")

# Manual failover to next provider
llm = router.next_provider()

# See chain
print(router.provider_names())  # ['keypool', 'openrouter', 'groq', 'google', 'cloudflare', 'local']
```


## Usage

### Two Modes

**Pipeline mode** (default) is a fixed research → analysis → validation → writing
chain. It suits questions that need sources, fact-checking and a written report.

**Team mode** (`--team`) is the general-purpose one. The orchestrator plans the
task into independent subtasks, the `Send` API runs one worker per subtask at the
same time, and a synthesizer merges them into a single answer. Use it for building
things, designs, comparisons, or anything with parts that can be worked on at once.

```bash
# Team mode: plan, fan out, synthesize
python -m src.main "Design a landing page for a specialty coffee shop" --team

# Cap the plan at 3 subtasks and run at most 2 workers at once
python -m src.main "Compare Postgres and SQLite for a local-first app" --team --subtasks 3 --workers 2
```

### How a claim is verified

Pipeline mode grades the run's claims before anyone writes them up. After the
researcher and the analyst finish, the validator makes one model call per pass
and asks for a verdict on every claim, with the text the claim's source actually
returned in the prompt:

- `supported` — the fetched source text states the claim
- `unsupported` — the source is silent, off-topic, or nothing was ever fetched
- `contradicted` — the source says something incompatible with the claim

The grade is load-bearing, not a report. The writer never sees a contradicted
claim, and its prompt says how many claims were removed for that reason.
Unsupported claims reach the writer labelled `UNVERIFIED`, and the finished
answer ends with a short provenance footer naming them. When the sources
contradict every finding, the graph does not run the writer at all: the run ends
saying it could not verify the evidence, instead of answering anyway.

A validator reply the parser cannot read, an empty reply, or a verdict call that
fails, grades as unverified — the failure mode is a missing grade, never a false
pass. The analysis pass is a judgement about the findings, not a self-reported
confidence score.

The cost is one extra small completion per validation pass: two in pipeline mode
(one for the findings, one for the analysis). Verdict calls ask the gateway to
skip reasoning — with it on, pooled reasoning models spend the whole completion
budget thinking and return no verdict at all. In a measured pipeline run the two
verdict calls were 4,066 prompt tokens and 208 completion tokens together,
about 3 seconds of a 31-second run.

### Threads remember

Run state is checkpointed to `multi-agent.sqlite`, so a second run in the same
thread continues the first one: earlier messages are still there, and the system
says so (`Resuming thread 'demo' with 6 earlier messages`).

```bash
multi-agent "The deploy target is a Windows PC named My_Pc" --thread demo --fresh
multi-agent "Which machine is the deploy target?" --thread demo
```

`--fresh` throws the thread's history away and starts clean. Set
`CHECKPOINT_DB=` (empty) in `.env` to keep everything in memory instead, or set
`DATABASE_URL` to use Postgres.

### Build mode: watch a page being written

```bash
multi-agent "A landing page for a neighbourhood coffee shop" --build --subtasks 4
```

Build mode plans the page into sections, writes each section as an HTML fragment
in parallel, and assembles them into one styled document. There is no synthesizer
call: assembly is deterministic (`src/artifacts.py`), which is what lets the page
be re-rendered after every fragment. In the dashboard you watch it happen: the planned
sections appear as placeholders, then fill with real content as each worker
finishes, and an agent cursor hovers over whichever section is being written
right now, labelled with the worker and its state.

### Live dashboard (chat)

On Windows, double-click **`start.bat`** — it checks for Python and `.env`, starts
the server, and opens the browser. From a terminal:

```bash
start.bat                        # or: multi-agent-dashboard
```

A chat interface over the agent team, laid out in three columns you can each
collapse: saved conversations on the left (with Settings at the bottom), the
conversation or the live page in the middle, and an Activity panel on the right
listing which agent is working right now plus every tool call as it happens.

- **Agent** — one agent that works the task end to end with tools, including the
  computer when it's enabled. Use it for "do this on my machine"; it does not plan
  a team, it just works
- **Team** — parallel subagents over a plan (CLI `--team`)
- **Research** — sourced answer. **Hold the button 1s** for *Deep research* (orange):
  the planner writes ~10 questions, a finder per question works them concurrently,
  a gap analyst names what is still missing, a second wave closes those gaps, then
  one report is written with sources
- **Build** — the middle column *becomes* the page being written. **Hold 1s** for
  *Super build* (red): more sections and each writer told to go all out: sections appear
  as placeholders, fill in as workers finish, and an agent cursor hovers over the
  one being written. The composer underneath stays live, so "make the hero
  bolder" is a follow-up that rebuilds it.

Follow-ups continue the same thread, and the workflow reads earlier turns from its
checkpointer, so memory survives messages and restarts (`chats.json` is the
visible log, `multi-agent.sqlite` the real state).

**Settings are editable.** ⚙ Settings in the sidebar changes the model, thinking
power, parallel workers, model retries, backoff, tool retries and the call timeout. Changes are
written to `.env` and applied to the next message — the model client is rebuilt on
save, so a new model id takes effect without a restart. Values out of range are
refused with the reason, and only that whitelist is writable: API keys and the
gateway URL are never exposed to the page.

**Leaving comments.** Press **Comment**, click any part of the UI, and write what
should change there. Notes are saved to `dashboard-notes.json` against the element
they were pinned to and listed under the Notes tab. That is the intended way to
ask for a layout change: mark it where it is wrong.

### Voice (fully local)

Speech-to-text runs on this machine with faster-whisper — no key, no quota, and no
audio leaves the computer. Install once:

```bash
pip install -e ".[voice]"
```

The model downloads on first use and is then cached. `medium` is the default and
runs on the GPU when there is one — on an RTX 3050 it transcribes a five-second
sentence in about 0.65s, where `small` on CPU took 3.6s. Settings has both the
model (base / small / distil-small.en / medium / large-v3) and the hardware
(auto / cuda / cpu); `auto` uses the GPU when CTranslate2 can see one and silently
falls back to CPU if a CUDA load fails.
Every transcription is appended to `logs/voice.log` with the clip size, how long
it took and what was heard, so a bad run can be diagnosed afterwards.

**Live call (🎧)** is the hands-free mode: it listens continuously, notices when you
stop talking, transcribes with Groq, runs the agent, speaks the answer aloud, and
goes straight back to listening. Talk over it and it stops mid-sentence to take
the new instruction, the way a person would. The bar shows what it is doing —
listening, hearing you, thinking, speaking — with a live level meter.

Saying "stop", "cancel" or "never mind" ends the turn instead of being treated as
a question. Replies can be read aloud with the
speaker button, using the browser's own speech engine — Groq's text-to-speech
models have been decommissioned, so nothing there is billed.

Everything is local, so there is nothing to configure and no key to lose.

### Live agent cursors

Add `--cursor` to see every agent working. Each agent gets its own on-screen
cursor and status chip, drawn in a click-through overlay that never takes focus
away from what you are doing, so a three-subtask team shows three cursors moving
at once.

```bash
python -m src.main "Design a landing page for a coffee shop" --team --cursor
```

The chip shows the agent's name and its current step — `thinking`,
`web_search: langgraph checkpointing`, `rate limited, retry in 6s`, `done`.

When an agent drives the desktop, the overlay also shows where the next action
lands. The pointer travels to the target, rings it, and names the action in plain
words — `Clicking Search`, `Typing into the address bar`, `Scrolling the page` —
then the ring fades and the chip goes two seconds later.

- **The ring** is a soft rounded outline around the control, the caret of the
  field being typed into, or the point being clicked. It pulses once as the
  cursor arrives and fades out after the action completes.
- **The chip** sits beside the ring with the agent's name over what it is doing,
  in words a person reads rather than tool names.
- **Escape** hides the overlay; the next action brings it back.
- **`CURSOR_GLIDE_MS`** (default `200`) is how long the travel takes, and
  **`CURSOR_SETTLE_MS`** (default `150`) holds the action back just long enough
  for the eye to follow it; `0` fires immediately.

The overlay is a no-op unless something turns it on: with the flag absent, with
`--no-cursor`, or with `CURSOR_OVERLAY` unset, a computer action pays about
0.6 µs and no thread is started. With it on, an action pays the settle delay plus
about 3 µs of publishing, and the drawing thread sits at roughly 0.4% of one
core. If the session has no usable desktop it prints
`cursor overlay unavailable: ...` once and carries on without it.

The glide, the ring, the chip placement and every fade are pure functions and
plain classes in `src/cursor_anim.py`, with no Tk import, so they are tested
without a display; `src/overlay.py` only renders what they decide.

### CLI

```bash
# Single shot
python -m src.main "Analyze Q3 AI market trends"

# Streaming mode
python -m src.main "Analyze Q3 AI market trends" --stream

# Custom thread
python -m src.main "Analyze AI agents" --thread my-session

# Different provider
python -m src.main "Analyze AI" --provider anthropic --model claude-3-5-sonnet-20241022
```

### Python API

```python
from src.graph.workflow import run_workflow
from src.config import get_llm

llm = get_llm()
result = await run_workflow("Your query here", thread_id="session-123", llm=llm)

print(result.final_report)
print(f"Findings: {len(result.research_findings)}")
print(f"Analysis: {len(result.analysis_results)}")
```

### Streaming

```python
from src.graph.workflow import run_workflow_stream

async for event in run_workflow_stream("Your query", thread_id="session-123"):
    if event.final_report:
        print("Report ready!")
    elif event.analysis_results:
        print(f"Analysis: {len(event.analysis_results)} insights")
```

## MCP Server (Tools)

```bash
# Start MCP server (stdio for local agents)
python -m src.mcp.server

# Start MCP server (HTTP for remote access)
python -m src.mcp.server --http
```

This server is how other AI agents — Claude Desktop, Claude Code, Hermes, or your
own agent loop — use this system as a tool. See [MCP.md](MCP.md) for the client
configuration.

`--http` binds `127.0.0.1:8001`. Listening on a network interface requires
`MCP_SERVER_HOST` **and** a shared `SERVER_AUTH_TOKEN`, and every request then
carries `Authorization: Bearer <token>`; the server refuses to start on a
non-loopback host without one.

Available tools:
- `research_report` - Run the full pipeline and return one cited report
- `team_task` - Plan, fan out to parallel subagents, and synthesize one answer
- `web_search` - Live DuckDuckGo results (title, URL, snippet)
- `fetch_url` - Fetch content from URL
- `search_github` - Search GitHub repositories, issues, or code (set `GITHUB_TOKEN` to raise the rate limit and enable code search)
- `get_current_time` - Get current time

## A2A Server (Remote Agents)

```bash
# Start A2A server
python -m src.a2a.server --http
```

Agent Card at: `http://127.0.0.1:8002/.well-known/agent.json`

Same bind rule as the MCP server: `A2A_SERVER_HOST` defaults to `127.0.0.1`, and a
network bind needs `SERVER_AUTH_TOKEN`, which every request must then present.

Skills:
- `research` - Gather information
- `analyze` - Analyze findings
- `write` - Generate reports
- `validate` - Validate findings
- `team` - Plan, run subagents in parallel, synthesize one answer
- `full_workflow` - Complete workflow

### Call Remote Agent

```python
from src.a2a.server import A2AAgentClient

client = A2AAgentClient("http://127.0.0.1:8002")  # add auth_token=... for a protected server
await client.discover()

result = await client.send_task("full_workflow", "Analyze AI market")
print(result)
```

## Verification Pipeline

```bash
# Run all verification gates
make verify

# Individual gates
make test       # Unit tests
make lint       # Ruff linting
make typecheck  # MyPy type checking
make sast       # Semgrep + Bandit security scan
```

## Evals: scoring the answers

`make test` proves the plumbing. The eval set scores what the plumbing produces:
fifteen cases across the three modes, each naming the properties a good answer
must have and the fabrications it must not contain.

```bash
make eval                  # scripted suite: no key, no gateway, no network
make eval ARGS=--live      # the fifteen cases against the KeyPool gateway
make eval ARGS="--live --case pipeline-stays-brief"   # one case
make eval ARGS=--list      # the cases, without running anything
```

How to read a row:

| Column | Reads |
| --- | --- |
| `score` | checks passed / checks asked, for that case |
| `checks` | how many passed, out of how many |
| `secs` | wall time for the case |
| `calls` | model responses, so a retry inside one call is not counted separately |
| `tokens` | prompt + completion tokens the provider reported |
| `verdicts` | claims graded supported/unsupported/contradicted by the validator |

A case passes only when every check passes; the summary score is the mean across
cases, so a case collapsing costs the same as another case improving. Failed
checks are listed under the table with what the check actually saw.

Checks read only what the system can establish about itself: the answer's text
(a required property present, a forbidden claim absent, a word cap, a source
link, a complete HTML document), the validator's verdicts, whether `abstain`
fired, how many subtasks were planned, and how many errors a run recorded. No
case needs the live internet to judge.

Every live run is written to `evals/last.json`. `--save-baseline` also stores it
as `evals/baseline.json`, and from then on each live run prints a per-case delta
and exits non-zero when any case scored lower - so a regression is a number
rather than an opinion. `--json PATH` writes the record somewhere else.

Stub mode (`--stub`, the default) runs three scripted scenarios through the real
pipeline graph: a grounded answer that must pass, a fabricated one that must
fail its exclusion, and a contradicted run that must abstain. CI runs it on every
push, which is how the gate stays credential-free.

What it does not measure: whether the prose is good, only whether the stated
properties hold; and each case is a single sample, so a borderline answer can
flip between runs. Re-run a case three times before believing a one-point move.

## Browser test: the dashboard end to end

The unit suite proves the pieces. `e2e/` proves the thing a person touches:
Chromium opens the dashboard, types a question, sends it, and the test asserts
the answer rendered, the cost line appeared under it, and no console error
fired. The three visible modes are each entered and run: Agent, Research, and
Build (which must fill the preview with a page).

```bash
python -m playwright install chromium   # once per machine
python -m pytest e2e -q                 # or: make browser-test
```

The model is scripted, so the test needs no gateway, no key and no network:
`DASHBOARD_FAKE_MODEL=1` makes the dashboard run turns on
`src/dashboard/fake_model.py`, which answers each agent's prompt from a fixed
script. Only the model is faked - the page, the HTTP routes, the SSE stream and
the graphs are the real ones.

CI runs it as its own `browser` job, apart from the unit matrix, so a broken tab
and a broken unit test never hide each other's results. If Chromium downloads
turn flaky there, run the same step inside the matching
`mcr.microsoft.com/playwright` image instead, which ships the browser.

What it does not cover: prose quality, real provider behaviour, voice, and
anything behind a microphone or a click the test does not make.

## Project Structure

```
multi-agent-system/
├── src/
│   ├── config.py              # Settings & LLM setup
│   ├── main.py                # CLI entry point
│   ├── observability.py       # OTel + LangSmith
│   ├── tools.py               # web_search, fetch_url, search_github, current_time
│   ├── activity.py            # the event bus: what each agent is doing now
│   ├── overlay.py             # live on-screen cursors, one per agent
│   ├── cursor_anim.py         # cursor glide, ring and fade timing (no Tk)
│   ├── dashboard/             # web dashboard (FastAPI + SSE + one HTML page)
│   ├── artifacts.py           # page assembly + shared stylesheet for build mode
│   ├── agents/                # Specialized agents
│   │   ├── base.py            # BaseAgent: retry, backoff, provider failover
│   │   ├── __init__.py        # Researcher, Analyst, Writer, Validator
│   │   └── team.py            # Planner, Worker, Synthesizer
│   ├── graph/                 # LangGraph workflows
│   │   ├── state.py           # TypedState with reducers
│   │   ├── workflow.py        # Pipeline graph
│   │   └── team.py            # Team graph (Send fan-out)
│   ├── mcp/                   # MCP server
│   │   └── server.py          # Tool server (stdio + HTTP)
│   └── a2a/                   # A2A protocol
│       └── server.py          # A2A server + client
├── evals/                     # Answer-quality evals
│   ├── cases.json             # The cases: properties an answer must have and must not
│   ├── run.py                 # python -m evals.run (stub suite or --live)
│   ├── harness.py             # Scoring, metrics, baseline comparison
│   └── stub.py                # Scripted model that proves the harness in CI
├── tests/
│   └── test_multi_agent.py    # Unit tests
├── e2e/                       # Browser test: the dashboard in Chromium
├── Makefile                   # Commands
├── pyproject.toml             # Dependencies
└── .env.example               # Environment template
```

## Computer use (optional)

Off by default. Install the extra and turn it on in Settings (or `COMPUTER_USE=true`):

```bash
pip install -e ".[computer]"
```

Agents then get `screen_describe` (open windows, the focused window's named
controls and their screen coordinates), `screen_shot`, `mouse_click`, `type_text`,
`press_key` and `scroll`. A text model cannot see a screenshot, so the written
description is what it works from, and every input action counts against
`COMPUTER_MAX_ACTIONS` for the turn. Screenshots land in `screenshots/`.

Each action names what it is aiming at, which is what the on-screen cursor points
at: `mouse_click(x, y, target="Search")`, `type_text(text, target="the address
bar")`. Typing and key actions ring the caret of the focused field, read straight
from Win32 with no accessibility-tree walk.

## Production Deployment

### Database (Required for Durability)

```bash
# Start PostgreSQL
docker run -d --name postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=multiagent \
  -p 5432:5432 postgres:16

# Update DATABASE_URL in .env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/multiagent
```

### OTel Collector (For Traces)

```bash
# Run OTel Collector
docker run -d --name otel-collector \
  -p 4317:4317 \
  -p 16686:16686 \
  otel/opentelemetry-collector-contrib:latest
```

### Docker

```bash
# Build
make docker-build

# Run
make docker-run QUERY="Analyze AI trends"
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | The provider every call goes to | `keypool` |
| `ENABLE_FREE_FALLBACK` | Add other providers as fallbacks | `false` |
| `KEYPOOL_BASE_URL` | Gateway base URL, including `/v1` | `http://127.0.0.1:8787` |
| `OPENAI_MODEL` | OpenAI model | `gpt-4o-mini` |
| `DATABASE_URL` | Postgres for the durable checkpointer | unset |
| `CHECKPOINT_DB` | SQLite file for run state | `multi-agent.sqlite` |
| `LANGSMITH_API_KEY` | LangSmith tracing | None |
| `OTEL_ENABLED` | Export traces/metrics to a collector | `false` |
| `OTEL_ENDPOINT` | OTel collector | `http://localhost:4317` |
| `GITHUB_TOKEN` | GitHub token for `search_github` | unset (5 req/min) |
| `DASHBOARD_PORT` | Dashboard port | `8765` |
| `LLM_TIMEOUT_SECONDS` | Hard cap on one model call | `120` |
| `MAX_FALLBACK_RETRIES` | Attempts per model call before giving up | `4` |
| `LLM_RETRY_BACKOFF` | Base seconds between retries | `3` |
| `TOOL_RETRIES` | Attempts per tool call | `2` |
| `THINKING_LEVEL` | Thinking power: off/low/medium/high/max | `medium` |
| `COMPUTER_USE` | Let agents drive the desktop | `false` |
| `COMPUTER_MAX_ACTIONS` | Input actions allowed per turn | `25` |
| `CURSOR_OVERLAY` | Show the live cursor overlay without `--cursor` | `false` |
| `CURSOR_SETTLE_MS` | Pause before each action so the eye can follow | `150` |
| `CURSOR_GLIDE_MS` | How long the cursor takes to travel to its target | `200` |
| `MCP_SERVER_HOST` | MCP server bind host | `127.0.0.1` |
| `MCP_SERVER_PORT` | MCP server port | `8001` |
| `A2A_SERVER_HOST` | A2A server bind host | `127.0.0.1` |
| `A2A_SERVER_PORT` | A2A server port | `8002` |
| `SERVER_AUTH_TOKEN` | Bearer token both HTTP servers require; mandatory for a non-loopback bind | unset |

## Why This Works

| Problem | Solution |
|---------|----------|
| Context loss at handoffs | Single shared state, no handoff payloads |
| Race conditions | `Annotated[list, operator.add]` reducers |
| Crash → lose progress | Postgres checkpointer every super-step |
| Silent hallucinations | Validator grades every claim against the source text fetched for it; contradicted claims never reach the writer, and a fully contradicted run declines to answer |
| Debugging = impossible | OTel spans: app→session→agent→trace→span |
| External agent integration | A2A protocol — looks like local tool |
| Tool schema breaking changes | MCP server versioning |

## License

MIT