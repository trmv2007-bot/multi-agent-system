"""A scripted model, so the harness can be tested without a credential.

Three scenarios run the real pipeline graph end to end: a grounded answer that
must pass, a fabricated answer that must fail the exclusion check, and a run
whose sources contradict every claim, which must abstain instead of writing.

If :func:`selftest` finds nothing wrong with how those three scored, then the
scorer, the metrics and the graph wiring all work - which is exactly what CI can
check without a key, a gateway, or the network.
"""

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from evals.harness import Case, CaseResult, Meter, run_case
from src.agents.verify import CONTRADICTED, SUPPORTED

RESEARCH = "Acme reported revenue of $1.1bn for the 2025 financial year."
GROUNDED = "Acme reported revenue of $1.1bn for the 2025 financial year."
FABRICATED = "Acme revenue hit $4.2bn in March 2026."
ANALYSIS = "Acme's 2025 revenue was $1.1bn."

QUESTION = "What was Acme's revenue in 2025?"

STUB_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class Scenario:
    """One scripted run and whether the harness should grade it a pass."""

    id: str
    verdict: str
    answer: str
    checks: dict[str, Any]
    should_pass: bool


def _checks(extra: dict[str, Any]) -> dict[str, Any]:
    """The baseline a scripted run must satisfy, plus whatever one scenario adds."""
    return {
        "answer_contains_any": [["$1.1bn"]],
        "answer_excludes": ["4\\.2bn"],
        "validation_required": True,
        "errors_max": 0,
        **extra,
    }


SCENARIOS = (
    Scenario(
        id="stub-grounded-answer",
        verdict=SUPPORTED,
        answer=GROUNDED,
        checks=_checks({"abstain": "forbidden"}),
        should_pass=True,
    ),
    Scenario(
        id="stub-fabrication-caught",
        verdict=SUPPORTED,
        answer=FABRICATED,
        checks=_checks({"abstain": "forbidden"}),
        should_pass=False,
    ),
    Scenario(
        id="stub-abstains-on-contradiction",
        verdict=CONTRADICTED,
        answer=FABRICATED,
        checks={
            "abstain": "required",
            "answer_contains_any": [["could not verify"]],
            "validation_required": True,
            "errors_max": 0,
        },
        should_pass=True,
    ),
)


class ScriptedModel:
    """Answers each pipeline prompt from a script, and never touches the network."""

    def __init__(self, verdict: str, answer: str, meter: Meter) -> None:
        self._verdict = verdict
        self._answer = answer
        self._meter = meter
        self.prompts: list[list[dict[str, Any]]] = []

    def bind(self, **_kwargs: Any) -> "ScriptedModel":
        """Accept request options; the script does not change with them."""
        return self

    def bind_tools(self, _tools: Any) -> "ScriptedModel":
        """Accept tools; the script never calls one."""
        return self

    async def ainvoke(self, messages: list[dict[str, Any]], **_kwargs: Any) -> Any:
        """Answer one prompt by role."""
        self.prompts.append(messages)
        self._meter.calls += 1
        return SimpleNamespace(
            content=self._reply(messages),
            tool_calls=None,
            response_metadata={},
            usage_metadata=None,
        )

    def _reply(self, messages: list[dict[str, Any]]) -> str:
        """Route on the marker each agent's prompt carries."""
        text = "\n".join(str(step.get("content", "")) for step in messages)
        if "Research this topic thoroughly" in text:
            return RESEARCH
        if "Check every claim below" in text:
            return self._verdicts(text.count("CLAIM "))
        if "Analyze these findings" in text:
            return ANALYSIS
        return self._answer

    def _verdicts(self, count: int) -> str:
        """One verdict per claim in the prompt."""
        return json.dumps(
            [
                {"index": index, "verdict": self._verdict, "reason": "scripted"}
                for index in range(max(count, 1))
            ]
        )


async def run_stub_suite() -> list[CaseResult]:
    """Run every scripted scenario through the real pipeline graph."""
    results: list[CaseResult] = []
    for scenario in SCENARIOS:
        case = Case(
            id=scenario.id,
            mode="pipeline",
            question=QUESTION,
            why="scripted scenario proving the harness discriminates",
            checks=scenario.checks,
        )
        meter = Meter()
        model = ScriptedModel(scenario.verdict, scenario.answer, meter)
        results.append(await run_case(case, model, meter, timeout=STUB_TIMEOUT_SECONDS))
    return results


def selftest(results: list[CaseResult]) -> list[str]:
    """What is wrong with the harness, given how the scripted runs scored."""
    problems: list[str] = []
    by_id = {result.id: result for result in results}
    for scenario in SCENARIOS:
        result = by_id.get(scenario.id)
        if result is None:
            problems.append(f"{scenario.id}: no result")
            continue
        expected = "pass" if scenario.should_pass else "fail"
        if result.passed != scenario.should_pass:
            problems.append(f"{scenario.id}: expected {expected}, scored {result.score:.2f}")
        if result.metrics["model_calls"] <= 0:
            problems.append(f"{scenario.id}: the graph made no model calls")
        verdicts = result.metrics["verdicts"]
        if sum(verdicts.values()) <= 0:
            problems.append(f"{scenario.id}: no claim carried a verdict")

    caught = by_id.get("stub-fabrication-caught")
    if caught is not None and not any(
        not check["passed"] and check["check"].startswith("answer_excludes")
        for check in caught.checks
    ):
        problems.append("the scorer did not catch a forbidden claim")

    abstained = by_id.get("stub-abstains-on-contradiction")
    if abstained is not None and not abstained.metrics["abstained"]:
        problems.append("the abstain path did not fire on a fully contradicted run")

    grounded = by_id.get("stub-grounded-answer")
    if grounded is not None and grounded.metrics["answer_chars"] <= 0:
        problems.append("the grounded run produced no answer")

    return problems
