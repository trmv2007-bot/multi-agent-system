# Multi-Agent System

An agent team that runs on your own machine. LangGraph for the graph and the
checkpointer, MCP for the tools, A2A to be called as a remote agent, and
OpenTelemetry for traces — with a local dashboard over all of it.

Five graphs share one typed state, one tool set and one retry policy. Pick the one
that matches the task:

| Mode | What actually runs | How to reach it |
| --- | --- | --- |
| **Pipeline** | researcher → analyst → the validator grades every claim → writer, or an honest abstain | the CLI default |
| **Team** | an orchestrator plans subtasks, LangGraph's `Send` runs one worker per subtask at once, a synthesizer merges them | `--team` |
| **Research** | a planner writes the questions, one finder per question, then one sourced report. Deep research adds a gap analyst and a second wave of finders | dashboard **Research** (hold for deep) |
| **Build** | outline → research → design → one writer per section → deterministic assembly → review → rewrite only the flagged sections | `--build`, dashboard **Build** |
| **Agent** | one agent working the task end to end with tools, and the desktop when it is enabled | dashboard **Agent** (the default) |

## See it run in one command, with no key

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e .
multi-agent-demo
```

It opens the dashboard at `http://127.0.0.1:8765`. Every model call is answered
from the script in [`src/demo.py`](src/demo.py), so the graph, the validator, the
checkpointer, the cost meter and the whole interface are real while the words are
scripted — nothing leaves the machine and nothing is billed. Ask it anything; try
**Research** and **Build** to watch the fan-out modes. On Windows,
`start-demo.bat` does the same thing on a double-click.

The CLI shows the same scripted run without a browser:

```bash
DEMO_MODE=true python -m src.main "What does this system do?" --fresh   # set DEMO_MODE=true on Windows cmd
```

## Quick start, with a real model

The system talks to an OpenAI-compatible endpoint. The default is a KeyPool
gateway on `127.0.0.1:8787`; any of the providers in `.env.example` works.

```bash
# 1. Dependencies — the same editable install as above, plus the dev tools
pip install -e ".[dev]"

# 2. Configuration: .env is read from the project root, wherever you run from
copy .env.example .env          # macOS/Linux: cp .env.example .env
#    then set KEYPOOL_API_KEY (or GROQ_API_KEY, OPENAI_API_KEY, ...)

# 3. Check the chain answers before spending a run
multi-agent --providers

# 4. First answer
python -m src.main "What changed about LangGraph checkpointing in 2026?"
```

`pip install -e .` puts `multi-agent`, `multi-agent-dashboard`,
`multi-agent-demo` and `multi-agent-mcp` on your PATH, so none of them depends on
the working directory.

**Or run the checks instead of an answer.** The suite is credential-free and takes
a few seconds, and `make` is optional — these are the commands behind it:

```bash
ruff check src/ tests/ evals/ e2e/          # lint
ruff format --check src/ tests/ evals/ e2e/ # formatting
mypy src/ evals/ e2e/                       # types
python -m pytest tests/ -q                  # 255 tests, no key, no network
python -m evals.run --stub                  # the eval harness, scripted
python -m pytest e2e -q                     # the dashboard in Chromium
```

`make lint`, `make typecheck`, `make test` and `make verify` run the same commands
if you have make; `make browser-test` is the browser suite.

## What verification actually checks

Pipeline mode does not take the researcher's word for anything. After research and
analysis, the validator grades every claim with one model call per pass, handing the
model the text the claim's source actually returned:

- `supported` — the fetched source text says it
- `unsupported` — the source is silent, off-topic, or nothing was ever fetched
- `contradicted` — the source says something incompatible with the claim

The grade decides the run rather than decorating it. A contradicted claim never
reaches the writer. An unsupported one reaches it labelled `UNVERIFIED`, and the
answer carries a provenance footer naming it. If the sources contradict *every*
claim, the graph takes the `abstain` edge and never runs the writer: the run ends by
saying it could not verify the evidence. A grade the parser cannot read counts as
unverified, so the failure mode is a missing grade, never a false pass.

**What it costs:** two extra small completions per pipeline run, one for the
findings and one for the analysis — in one measured run 4,066 prompt and 208
completion tokens together, about 3 seconds of a 31-second run. Verdict calls ask
the gateway to skip reasoning, because a pooled reasoning model otherwise spends the
whole completion budget thinking and returns no verdict at all. The mechanical detail
is under [How a claim is verified](#how-a-claim-is-verified).

## What a run costs

Every run ends with one line, and the same numbers ride along in the dashboard:

```
Cost:              5 model calls · 1.8k tokens (1.6k in, 212 out) · 0.1s · slowest Validator
```

That is model calls (not retries — a retry inside a call is still one call), the
tokens the provider reported, wall time, the slowest agent, and a count of JSON
repairs and JSON failures when either is non-zero. `evals/last.json` records the same
figures per case.

## Evals: what the answers are worth

`python -m pytest` proves the plumbing. [`evals/cases.json`](evals/cases.json) scores
what the plumbing produces: fifteen cases across pipeline, team and build, each
naming properties a good answer must have and the fabrications it must not contain.

```bash
python -m evals.run                    # scripted suite: no key, no gateway, no network
python -m evals.run --list             # the cases and why each one exists
python -m evals.run --live             # the fifteen cases against your gateway
python -m evals.run --live --case pipeline-stays-brief
```

A row reads: `score` is checks passed / checks asked, `calls` is model responses,
`secs` the wall time, `verdicts` the validator's tally. A case passes only when every
check passes, and a failed check is listed with what it actually saw. A case where no
model answered prints `unmeasured` instead of a score and is never counted as a
quality figure — a dead gateway and a bad answer are different failures, and the exit
code says which one you have. The scripted
suite runs three scenarios through the real graph — a grounded answer that must pass,
a fabricated one that must fail, and a contradicted one that must abstain — and
selftests that the scorer discriminated between them. That is the piece CI runs on
every push, which is why the gate stays credential-free. Live runs are written to
`evals/last.json`, and `evals/baseline.json` is committed, so every `--live` run
compares itself against that baseline and exits non-zero on a per-case regression —
a regression is a number rather than an opinion. `--baseline PATH` compares against
another run, `--save-baseline` replaces the committed one, and `--pause` spaces the
cases out on a rate-limited gateway. Details, including what the checks cannot see,
are under [Evals: scoring the answers](#evals-scoring-the-answers).

## Watch the agents work

Add `--cursor` to a run and every agent appears on screen: its own cursor, its own
status chip with the current step (`thinking`, `web_search: langgraph checkpointing`,
`done`). A three-subtask team draws three cursors.

```bash
python -m src.main "A landing page for a neighbourhood coffee shop" --build --cursor
```

Each action names what it aims at — `mouse_click(x, y, target="Search")`,
`type_text(text, target="the address bar")` — so the pointer travels to the target,
rings it, and names the action in words rather than tool names. Escape hides the
overlay; the next action brings it back. `CURSOR_GLIDE_MS` and `CURSOR_SETTLE_MS`
set the travel and settle timings. With the flag absent and `CURSOR_OVERLAY` unset,
the overlay is a no-op.

## Architecture

Pipeline mode, the CLI default — including the `abstain` edge, which is the graph's
one honest refusal:

```mermaid
flowchart TD
    S([start]) --> R["researcher<br/>web_search · fetch_url · gather_sources"]
    R --> A["analyst"]
    R --> VF["validate findings<br/>one verdict per claim, against the fetched text"]
    A --> VA["validate analysis"]
    VF --> VA
    VA -->|"not every claim contradicted"| W["writer<br/>never sees a contradicted claim"]
    VA -->|"every claim contradicted"| X["abstain<br/>declines to answer"]
    W --> E([answer])
    X --> E
```

Team and research modes fan out with the `Send` API and fan back in:

```mermaid
flowchart LR
    S([start]) --> P["plan"]
    P -->|"Send, one per subtask"| W1["worker 1"]
    P --> W2["worker 2"]
    P --> WN["worker n"]
    W1 --> Y["synthesize"]
    W2 --> Y
    WN --> Y
    Y --> E([answer])
```

```mermaid
flowchart LR
    R([start]) --> Q["plan questions"] --> F["finder per question"] --> P["report"]
    P --> RE([sourced answer])
```

Deep research inserts the gap pass: `find → gaps → (refine | report)`, where `refine`
is the same finder node with the questions the gap analyst found missing.

Build mode is a loop rather than a single pass — the review rewrites only the
sections it flagged, then the page is reassembled:

```mermaid
flowchart TD
    B0([start]) --> B1["plan sections"] --> B2["research"] --> B3["designer<br/>theme"] --> B4["replan"]
    B4 -->|"Send, one per section"| B5["section writers"]
    B5 --> B6["assemble page"]
    B6 --> B7["review"]
    B7 -->|"flagged sections"| B8["rewrite those sections"] --> B6
    B7 -->|"nothing flagged"| B9["report"] --> BE([page])
```

Agent mode is the small one: one node, one agent, as many tool rounds as the task
needs, then an answer.

Every mode shares `src/graph/state.py` (`AgentState` with reducers for the parallel
merges), per-turn scoping so a resumed thread only shows the current turn's findings,
and the checkpointer in `src/graph/workflow.py` — SQLite by default, Postgres if you
set `DATABASE_URL`, in memory if you clear `CHECKPOINT_DB`.

## Local Mode (Ollama, LM Studio, LocalAI)

To use a local LLM instead of KeyPool or cloud APIs:

```bash
# Edit .env to use local provider
LLM_PROVIDER=local
LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
LOCAL_LLM_MODEL=llama3.2

# Run
python -m src.main "Analyze AI agents"      # or: make run QUERY="Analyze AI agents"
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

### Choosing a mode

The table at the top of this file is the whole map. On the CLI:

```bash
# Pipeline (the default): research, grade every claim, then write
python -m src.main "Analyze the current state of AI agents in 2026"

# Team: plan, fan out, synthesize
python -m src.main "Design a landing page for a specialty coffee shop" --team

# Cap the plan at 3 subtasks and run at most 2 workers at once
python -m src.main "Compare Postgres and SQLite for a local-first app" --team --subtasks 3 --workers 2

# Build: sections in parallel, then review and repair
python -m src.main "A landing page for a neighbourhood coffee shop" --build --subtasks 4
```

`--stream` prints progress as each step lands instead of only the end, and
`--subtasks`/`--workers` set the plan ceiling and the concurrency.

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

- **Agent** (the default) — one agent that works the task end to end with tools,
  including the computer when it's enabled. Use it for "do this on my machine"; it
  does not plan a team, it just works
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

**Answers are markdown, and they render like it.** Bold, italic, tables with
alignment, fenced code with its language, nested lists, blockquotes, headings and
links all arrive as markup rather than as asterisks and pipes. The renderer is
vendored in `src/dashboard/static/markdown.js` and served with the page, so it
needs no CDN and works with no network. Model output is escaped and sanitised
before it reaches the DOM: raw HTML in an answer shows as text, and a
`javascript:` link degrades to plain text. A block that arrives half-written while
the answer streams renders as code or as a padded row, never as a broken layout.

The page has buttons for Agent, Research and Build. Team mode and the pipeline are
CLI modes: the dashboard server accepts `"mode": "team"` and `"mode": "pipeline"` on
`POST /api/chat`, but nothing in the interface selects them.

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

# Different provider and model (the --provider choices are the router's chain)
python -m src.main "Analyze AI" --provider groq --model llama-3.3-70b-versatile
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
- `gather_sources` - Search, open the promising pages, and keep only the ones on topic
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

The gates CI runs, as commands — make is convenient, not required:

```bash
ruff check src/ tests/ evals/ e2e/           # lint
ruff format --check src/ tests/ evals/ e2e/  # formatting
mypy src/ evals/ e2e/                        # types
python -m pytest tests/ -q                   # 255 tests, no credentials
python -m evals.run --stub                   # the eval harness
bandit -c pyproject.toml -r src/             # SAST

# Optional: Semgrep (needs WSL on Windows)
semgrep scan --config=p/security-audit src/
```

CI runs those, plus `python -m pytest e2e -q` in a separate job with Chromium
installed — see [Browser test](#browser-test-the-dashboard-end-to-end).

`make verify` chains the same gates, and `make test`, `make lint`,
`make typecheck` and `make sast` run them one at a time.

## Evals: scoring the answers

`python -m pytest` proves the plumbing. The eval set scores what the plumbing
produces: fifteen cases across pipeline, team and build, each naming the properties a
good answer must have and the fabrications it must not contain.

```bash
python -m evals.run                  # scripted suite: no key, no gateway, no network
python -m evals.run --live           # the fifteen cases against the KeyPool gateway
python -m evals.run --list           # the cases, without running anything
python -m evals.run --live --case pipeline-stays-brief   # one case
```

`make eval` and `make eval ARGS="--live --case pipeline-stays-brief"` wrap the same
commands.

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
the answer rendered — as a `<strong>`, a real `<table>` with its cells, and a
`<pre><code>` with its language, with no `**` or `| --- |` left on screen — the
cost line appeared outside the answer, and no console error fired. The three visible modes are each entered and run: Agent, Research, and
Build (which must fill the preview with a page).

```bash
python -m playwright install chromium   # once per machine
python -m pytest e2e -q                 # or: make browser-test
```

The model is scripted, so the test needs no gateway, no key and no network:
`DASHBOARD_FAKE_MODEL=1` makes the dashboard answer from the same script the demo
runs, [`src/demo.py`](src/demo.py) — `src/dashboard/fake_model.py` is only the switch
and the answer text the test asserts on, so the test and the demo cannot drift apart.
Only the model is faked - the page, the HTTP routes, the SSE stream and the graphs
are the real ones.

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
│   ├── config.py              # Settings, providers, the failover chain
│   ├── main.py                # CLI entry point
│   ├── demo.py                # Scripted provider: the dashboard with no key
│   ├── observability.py       # OTel + LangSmith + the per-run cost meter
│   ├── structured.py          # call_json(): the one parse/repair/retry path
│   ├── tools.py               # web_search, fetch_url, search_github, gather_sources
│   ├── activity.py            # the event bus: what each agent is doing now
│   ├── overlay.py             # live on-screen cursors, one per agent
│   ├── cursor_anim.py         # cursor glide, ring and fade timing (no Tk)
│   ├── computer.py            # screen description and input actions (optional)
│   ├── voice.py               # local speech-to-text (optional)
│   ├── server_guard.py        # loopback bind + bearer token for the HTTP servers
│   ├── dashboard/             # web dashboard (FastAPI + SSE + one HTML page)
│   │   ├── fake_model.py      # the switch the browser test uses to script a run
│   │   └── static/            # index.html, and markdown.js which renders answers
│   ├── artifacts.py           # page assembly + shared stylesheet for build mode
│   ├── agents/                # Specialized agents
│   │   ├── base.py            # BaseAgent: retry, backoff, provider failover
│   │   ├── __init__.py        # Researcher, Analyst, Writer, Validator
│   │   ├── verify.py          # claim grading: the verdicts and the abstain rule
│   │   ├── team.py            # Planner, Worker, Synthesizer
│   │   ├── research.py        # Question planner, finder, gap analyst, reporter
│   │   ├── build.py           # Page planner, designer, section writer, critic
│   │   └── contract.py        # the writer's own output contract
│   ├── graph/                 # LangGraph workflows
│   │   ├── state.py           # AgentState with reducers
│   │   ├── workflow.py        # Pipeline graph + checkpointer scope
│   │   ├── team.py            # Team graph (Send fan-out)
│   │   ├── research.py        # Research graph (two waves)
│   │   ├── build.py           # Build graph (write, review, repair)
│   │   └── agent.py           # Agent graph (one agent, many tool rounds)
│   ├── mcp/                   # MCP server
│   │   └── server.py          # Tool server (stdio + HTTP)
│   └── a2a/                   # A2A protocol
│       └── server.py          # A2A server + client
├── evals/                     # Answer-quality evals
│   ├── cases.json             # The cases: properties an answer must have and must not
│   ├── run.py                 # python -m evals.run (stub suite or --live)
│   ├── harness.py             # Scoring, metrics, baseline comparison
│   ├── baseline.json          # The committed baseline every live run is compared to
│   └── stub.py                # Scripted scenarios that prove the harness in CI
├── tests/                     # 263 tests, no credentials, no network
├── e2e/                       # Browser test: the dashboard in Chromium
├── start.bat                  # Double-click the dashboard (needs .env)
├── start-demo.bat             # Double-click the demo dashboard (needs nothing)
├── Makefile                   # Convenience targets over the commands above
├── pyproject.toml             # Dependencies + console scripts
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

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | The provider every call goes to | `keypool` |
| `DEMO_MODE` | Answer every model call from `src/demo.py` instead | `false` |
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