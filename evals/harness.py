"""Score the quality of the system's answers against a fixed set of cases.

``tests/`` proves the plumbing works. This scores the answers: each case names
the properties a good answer must have and the fabrications it must not
contain, and every check reads something the system can establish about itself -
a validator verdict, whether ``abstain`` fired, the text the run produced, and
what it cost.

Two ways to run it: the live suite (needs the gateway) and the scripted suite in
:mod:`evals.stub`, which proves the scorer itself without a credential.
"""

import asyncio
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from src.activity import activity
from src.agents.verify import CONTRADICTED, SUPPORTED, UNSUPPORTED, verdicts_of
from src.graph.build import run_build
from src.graph.state import AgentState
from src.graph.team import run_team
from src.graph.workflow import run_workflow

EVALS_DIR = Path(__file__).resolve().parent
CASES_FILE = EVALS_DIR / "cases.json"
LAST_FILE = EVALS_DIR / "last.json"
BASELINE_FILE = EVALS_DIR / "baseline.json"

MODES = ("pipeline", "team", "build")

CHECK_KEYS = frozenset(
    {
        "answer_min_words",
        "answer_max_words",
        "answer_contains_any",
        "answer_matches",
        "answer_excludes",
        "answer_requires_link",
        "abstain",
        "findings_min",
        "validation_required",
        "plan_min",
        "sections_required",
        "page",
        "errors_max",
    }
)
"""Every check a case may ask for. An unknown key is a typo, not a silent skip."""

BUILD_SECTIONS = 4
"""Sections build mode may plan for an eval case: enough for a real page."""

EVAL_MAX_TOKENS = 4096
"""Token budget per call when a caller does not set one.

The pooled gateway routes to reasoning models that spend a small budget on
reasoning and return empty content with ``finish_reason="length"``. A generous
cap plus ``reasoning_effort: none`` (see :func:`live_model`) is what makes a
measured run produce text instead of silence.
"""

CASE_TIMEOUT_SECONDS = 300.0
"""Wall-clock cap on one case, so one stalled gateway call cannot hang the suite."""

UNREACHABLE = (
    "could not reach a model",
    "http 429",
    "429",
    "rate limit",
    "empty content",
    "timed out",
    "connection",
)
"""What a failure looks like when the model never answered.

A pooled gateway running out of keys is not a bad answer. Scoring it as one
would report a quality collapse that never happened, and a baseline taken during
one would be worthless, so these cases are reported as unmeasured instead.
"""


@dataclass(frozen=True)
class Case:
    """One task, the properties its answer needs, and why it is worth asking."""

    id: str
    mode: str
    question: str
    why: str
    checks: dict[str, Any]


def load_cases(path: Path | None = None) -> list[Case]:
    """Read the case file, refusing anything the scorer could not grade."""
    payload = json.loads((path or CASES_FILE).read_text(encoding="utf-8"))
    cases: list[Case] = []
    seen: set[str] = set()
    for entry in payload["cases"]:
        case = Case(
            id=str(entry["id"]),
            mode=str(entry["mode"]),
            question=str(entry["question"]),
            why=str(entry.get("why", "")),
            checks=dict(entry.get("checks") or {}),
        )
        if case.mode not in MODES:
            raise ValueError(f"{case.id}: unknown mode {case.mode!r}, expected one of {MODES}")
        unknown = set(case.checks) - CHECK_KEYS
        if unknown:
            raise ValueError(f"{case.id}: unknown checks {sorted(unknown)}")
        if case.id in seen:
            raise ValueError(f"duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    return cases


@dataclass
class Meter:
    """What one case spent on the model."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    empty_replies: int = 0

    @property
    def tokens(self) -> int:
        """Prompt plus completion tokens reported by the provider."""
        return self.prompt_tokens + self.completion_tokens

    def record(self, response: Any) -> None:
        """Count one call and whatever usage it reported."""
        self.calls += 1
        prompt, completion = usage_of(response)
        self.prompt_tokens += prompt
        self.completion_tokens += completion


class EmptyReply(RuntimeError):
    """The model answered with no text, which is a failure and never a pass."""


class MeteredModel:
    """A chat client that counts every call and refuses an empty reply.

    It delegates binding, so an agent's ``bind(max_tokens=...)`` and
    ``bind_tools(...)`` still work; the options accumulate instead of replacing
    what the eval set already asked for.
    """

    def __init__(self, client: Any, meter: Meter, options: dict[str, Any] | None = None) -> None:
        self._client = client
        self._meter = meter
        self._options = dict(options or {})

    def bind(self, **kwargs: Any) -> "MeteredModel":
        """Accumulate request options, merging ``extra_body`` rather than replacing it."""
        extra_body = {
            **(self._options.get("extra_body") or {}),
            **(kwargs.pop("extra_body", None) or {}),
        }
        options = {**self._options, **kwargs}
        if extra_body:
            options["extra_body"] = extra_body
        options.setdefault("max_tokens", EVAL_MAX_TOKENS)
        return MeteredModel(self._client, self._meter, options)

    def bind_tools(self, tools: Any) -> "MeteredModel":
        """Bind tools, keeping the meter and the request options."""
        return MeteredModel(self._client.bind_tools(tools), self._meter, self._options)

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        """Call the model, count it, and fail loudly on an empty reply."""
        client = self._client.bind(**self._options) if self._options else self._client
        response = await client.ainvoke(messages, **kwargs)
        self._meter.record(response)
        if not text_of(response).strip() and not getattr(response, "tool_calls", None):
            self._meter.empty_replies += 1
            raise EmptyReply("the model returned empty content")
        return response


def live_model(llm: Any, meter: Meter) -> MeteredModel:
    """Wrap the gateway client with the body fields this gateway needs."""
    return MeteredModel(llm, meter, {"extra_body": {"reasoning_effort": "none"}})


def text_of(value: Any) -> str:
    """Model output as plain text, whether it came as a string or parts."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", "")) for part in content
        )
    return str(content or "")


def usage_of(response: Any) -> tuple[int, int]:
    """Prompt and completion tokens a response reports, or zeros."""
    usage = getattr(response, "usage_metadata", None) or {}
    if usage:
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    metadata = getattr(response, "response_metadata", None) or {}
    tokens = metadata.get("token_usage") or metadata.get("usage") or {}
    return int(tokens.get("prompt_tokens") or 0), int(tokens.get("completion_tokens") or 0)


@dataclass
class CaseResult:
    """One case's grade and what it cost to reach it."""

    id: str
    mode: str
    score: float
    passed: bool
    checks: list[dict[str, Any]]
    metrics: dict[str, Any]
    status: str = "graded"

    def as_dict(self) -> dict[str, Any]:
        """The JSON record the runner persists."""
        return {
            "id": self.id,
            "mode": self.mode,
            "status": self.status,
            "score": round(self.score, 3),
            "passed": self.passed,
            "checks": self.checks,
            "metrics": self.metrics,
        }


async def evaluate(
    cases: Sequence[Case],
    make_model: Callable[[Meter], Any],
    timeout: float = CASE_TIMEOUT_SECONDS,
    after_each: Callable[[list[CaseResult]], None] | None = None,
    pause: float = 0.0,
) -> list[CaseResult]:
    """Run every case in turn and grade it.

    Sequential on purpose: one case at a time keeps a pooled gateway readable
    and makes each case's calls and tokens attributable to it. ``after_each``
    sees the results so far after each case, so a sweep cut short on a dropping
    connection leaves a usable partial record instead of nothing.
    """
    results: list[CaseResult] = []
    for case in cases:
        meter = Meter()
        results.append(await run_case(case, make_model(meter), meter, timeout=timeout))
        if after_each is not None:
            after_each(results)
        if pause and len(results) < len(cases):
            # A pooled gateway rate-limits by how fast keys are spent, not by
            # how many cases there are, so a paced run measures more of them.
            await asyncio.sleep(pause)
    return results


async def run_case(case: Case, model: Any, meter: Meter, timeout: float) -> CaseResult:
    """Drive one case through its mode's graph, then score what came back."""
    thread = f"eval-{case.id}"
    activity.reset()
    started = time.monotonic()
    state: AgentState | None = None
    error: str | None = None
    try:
        state = await asyncio.wait_for(_drive(case, model, thread), timeout)
    except TimeoutError:
        error = f"timed out after {timeout:.0f}s"
    except Exception as exc:  # a case fails on its own error, never the suite's
        error = f"{type(exc).__name__}: {exc}"
    seconds = time.monotonic() - started

    checks = score_case(case, state, error)
    earned = sum(1 for check in checks if check["passed"])
    status = status_of(state, error, meter)
    return CaseResult(
        id=case.id,
        mode=case.mode,
        score=earned / len(checks) if checks and status != UNAVAILABLE_STATUS else 0.0,
        passed=status == "graded" and bool(checks) and earned == len(checks),
        checks=checks,
        metrics=metrics_of(state, meter, seconds),
        status=status,
    )


UNAVAILABLE_STATUS = "unavailable"
"""The case never got a model, so it has no score."""


def status_of(state: AgentState | None, error: str | None, meter: Meter) -> str:
    """Whether a case was graded, or whether no model was available to grade."""
    if error is not None:
        return UNAVAILABLE_STATUS if _unreachable(error) else "error"
    if meter.calls == 0:
        return UNAVAILABLE_STATUS
    answered = (state.final_report if state else "") or ""
    if state is not None and not answered.strip() and state.errors:
        if all(_unreachable(str(item.get("error", ""))) for item in state.errors):
            return UNAVAILABLE_STATUS
    return "graded"


def _unreachable(message: str) -> bool:
    """Whether a failure message describes a model that never answered."""
    lowered = message.lower()
    return any(marker in lowered for marker in UNREACHABLE)


async def _drive(case: Case, model: Any, thread_id: str) -> AgentState:
    """Run the case through the graph its mode names."""
    saver = MemorySaver()
    if case.mode == "team":
        return await run_team(case.question, thread_id, llm=model, checkpointer=saver, fresh=True)
    if case.mode == "build":
        return await run_build(
            case.question,
            thread_id,
            llm=model,
            max_sections=BUILD_SECTIONS,
            checkpointer=saver,
            fresh=True,
        )
    return await run_workflow(case.question, thread_id, llm=model, checkpointer=saver, fresh=True)


def score_case(case: Case, state: AgentState | None, error: str | None) -> list[dict[str, Any]]:
    """Grade one finished case against the properties it asked for."""
    if state is None:
        return [{"check": "run", "passed": False, "detail": error or "the run produced no state"}]
    return _answer_checks(case, state.final_report or "") + _run_checks(case, state)


def _answer_checks(case: Case, answer: str) -> list[dict[str, Any]]:
    """The checks that read the answer's own text."""
    words = len(answer.split())
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    floor = case.checks.get("answer_min_words")
    if floor is not None:
        add("answer_min_words", words >= int(floor), f"{words} words, need at least {floor}")
    ceiling = case.checks.get("answer_max_words")
    if ceiling is not None:
        add("answer_max_words", words <= int(ceiling), f"{words} words, allowed at most {ceiling}")
    for index, group in enumerate(case.checks.get("answer_contains_any") or [], start=1):
        add(
            f"answer_contains_any#{index}",
            _group_hit(group, answer) is not None,
            f"none of {group}",
        )
    for index, pattern in enumerate(case.checks.get("answer_matches") or [], start=1):
        add(
            f"answer_matches#{index}",
            _search(pattern, answer) is not None,
            f"nothing matched {pattern!r}",
        )
    for index, pattern in enumerate(case.checks.get("answer_excludes") or [], start=1):
        hit = _search(pattern, answer)
        add(f"answer_excludes#{index}", hit is None, f"matched {hit!r}")
    if case.checks.get("answer_requires_link"):
        add("answer_requires_link", _search(r"https?://\S+", answer) is not None, "no URL")
    for index, group in enumerate(case.checks.get("sections_required") or [], start=1):
        add(f"sections_required#{index}", _group_hit(group, answer) is not None, f"none of {group}")
    if case.checks.get("page"):
        lower = answer.lower()
        add("page", "<!doctype html" in lower and "</html>" in lower, "not a complete document")
    return checks


def _run_checks(case: Case, state: AgentState) -> list[dict[str, Any]]:
    """The checks that read the run's own record: verdicts, shape, errors."""
    verdicts = verdicts_of(state.validation_results, state.turn)
    abstained = state.context.get("verification") == "abstained"
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    if "abstain" in case.checks:
        add(*_abstain_check(str(case.checks["abstain"]), abstained))
    if case.checks.get("validation_required"):
        add("validation_required", bool(verdicts), "no claim was graded in this run")
    floor = case.checks.get("findings_min")
    if floor is not None:
        found = len([f for f in state.research_findings if f.get("turn") == state.turn])
        add("findings_min", found >= int(floor), f"{found} findings, need at least {floor}")
    floor = case.checks.get("plan_min")
    if floor is not None:
        add("plan_min", len(state.plan) >= int(floor), f"{len(state.plan)} planned, need {floor}")
    ceiling = case.checks.get("errors_max")
    if ceiling is not None:
        add(
            "errors_max",
            len(state.errors) <= int(ceiling),
            f"{len(state.errors)} errors, allowed {ceiling}",
        )
    return checks


def _group_hit(group: list[str], text: str) -> str | None:
    """The first phrase in one alternatives group that the text contains."""
    lowered = text.lower()
    return next((phrase for phrase in group if phrase.lower() in lowered), None)


def _abstain_check(wanted: str, abstained: bool) -> tuple[str, bool, str]:
    """Grade whether the run refused to answer, as the case asked."""
    if wanted == "required":
        return "abstain", abstained, "the run answered instead of abstaining"
    if wanted == "forbidden":
        return "abstain", not abstained, "the run abstained" if abstained else ""
    return "abstain", True, "abstained" if abstained else "answered"


def _search(pattern: str, text: str) -> str | None:
    """The matched text for one pattern with its surroundings, or None.

    The surrounding text is what makes a failed exclusion diagnosable: a bare
    match says a forbidden claim appeared, the window says whether it was the
    model's copy or the page's own script.
    """
    match = re.search(pattern, text, re.S)
    if match is None:
        return None
    start = max(0, match.start() - 40)
    return " ".join(text[start : match.end() + 40].split())


def verdict_counts(state: AgentState | None) -> dict[str, int]:
    """How this run's claims were graded."""
    counts = {SUPPORTED: 0, UNSUPPORTED: 0, CONTRADICTED: 0}
    if state is None:
        return counts
    for record in state.validation_results:
        if record.get("turn") != state.turn:
            continue
        verdict = str(record.get("verdict", ""))
        if verdict in counts:
            counts[verdict] += 1
    return counts


def metrics_of(state: AgentState | None, meter: Meter, seconds: float) -> dict[str, Any]:
    """What one case cost and what it produced."""
    answer = state.final_report if state is not None else ""
    return {
        "seconds": round(seconds, 1),
        "model_calls": meter.calls,
        "tokens": meter.tokens,
        "prompt_tokens": meter.prompt_tokens,
        "completion_tokens": meter.completion_tokens,
        "empty_replies": meter.empty_replies,
        "answer_words": len((answer or "").split()),
        "answer_chars": len(answer or ""),
        "answer_excerpt": " ".join((answer or "").split())[:600],
        "abstained": bool(state is not None and state.context.get("verification") == "abstained"),
        "verdicts": verdict_counts(state),
        "errors": [str(item.get("error", ""))[:160] for item in (state.errors if state else [])][
            :5
        ],
    }


def summarize(results: Sequence[CaseResult]) -> dict[str, Any]:
    """The one-line verdict for a whole run."""
    graded = [result for result in results if result.status == "graded"]
    verdicts = {SUPPORTED: 0, UNSUPPORTED: 0, CONTRADICTED: 0}
    for result in graded:
        for verdict, count in result.metrics["verdicts"].items():
            verdicts[verdict] = verdicts.get(verdict, 0) + count
    return {
        "score": round(sum(r.score for r in graded) / len(graded), 3) if graded else 0.0,
        "cases": len(graded),
        "measured": len(results),
        "passed": sum(1 for result in graded if result.passed),
        "failed": [result.id for result in graded if not result.passed],
        "unmeasured": [result.id for result in results if result.status != "graded"],
        "seconds": round(sum(result.metrics["seconds"] for result in results), 1),
        "model_calls": sum(result.metrics["model_calls"] for result in results),
        "tokens": sum(result.metrics["tokens"] for result in results),
        "verdicts": verdicts,
    }


def report_payload(results: Sequence[CaseResult], mode: str, model: str) -> dict[str, Any]:
    """The record written to ``last.json`` and to a saved baseline."""
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "model": model,
        "summary": summarize(results),
        "results": [result.as_dict() for result in results],
    }


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    """Write a run record where the next comparison can find it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def compare(results: Sequence[CaseResult], baseline: dict[str, Any]) -> dict[str, Any]:
    """Per-case score deltas against a saved run, with the regressions named.

    The overall figure is computed over the cases both runs hold, so comparing a
    filtered subset against a full baseline cannot invent a change.
    """
    then = {
        str(item.get("id")): float(item.get("score", 0.0))
        for item in baseline.get("results", [])
        if item.get("status", "graded") == "graded"
    }
    rows: list[dict[str, Any]] = []
    regressed: list[str] = []
    for result in results:
        before = then.get(result.id)
        row: dict[str, Any] = {
            "id": result.id,
            "then": None if before is None else round(before, 3),
            "now": round(result.score, 3),
            "delta": None,
        }
        if before is not None and result.status == "graded":
            row["delta"] = round(result.score - before, 3)
            if result.score < before:
                regressed.append(result.id)
        rows.append(row)

    shared = [row for row in rows if row["delta"] is not None]
    return {
        "rows": rows,
        "regressed": regressed,
        "compared": len(shared),
        "overall_then": round(sum(row["then"] for row in shared) / len(shared), 3)
        if shared
        else None,
        "overall_now": round(sum(row["now"] for row in shared) / len(shared), 3)
        if shared
        else None,
    }
