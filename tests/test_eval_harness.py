"""The eval harness must discriminate, or its scores are decoration.

Everything here uses the scripted model only: no key, no gateway, no network.
"""

from typing import Any

from evals import harness
from evals.harness import Case, CaseResult
from evals.stub import run_stub_suite, selftest
from src.graph.state import AgentState


def case(**checks: Any) -> Case:
    """One throwaway case carrying the checks under test."""
    return Case(id="unit", mode="pipeline", question="q", why="unit test", checks=checks)


def result(case_id: str, score: float) -> CaseResult:
    """A finished case result sized for a comparison."""
    return CaseResult(
        id=case_id,
        mode="pipeline",
        score=score,
        passed=score == 1.0,
        checks=[],
        metrics={
            "seconds": 1.0,
            "model_calls": 1,
            "tokens": 10,
            "verdicts": {"supported": 0, "unsupported": 0, "contradicted": 0},
        },
    )


class TestScriptedSuite:
    """A fake model drives the real graph, and the scorer grades what it produced."""

    async def test_the_harness_discriminates(self):
        results = await run_stub_suite()

        assert selftest(results) == []
        assert len(results) == 3

    async def test_a_grounded_answer_scores_full_marks(self):
        by_id = {item.id: item for item in await run_stub_suite()}

        assert by_id["stub-grounded-answer"].score == 1.0
        assert by_id["stub-grounded-answer"].metrics["model_calls"] > 0

    async def test_a_fabricated_answer_fails_its_exclusion(self):
        by_id = {item.id: item for item in await run_stub_suite()}
        failed = {
            check["check"]
            for check in by_id["stub-fabrication-caught"].checks
            if not check["passed"]
        }

        assert "answer_excludes#1" in failed
        assert by_id["stub-fabrication-caught"].score < 1.0

    async def test_a_contradicted_run_abstains_and_is_graded_as_one(self):
        by_id = {item.id: item for item in await run_stub_suite()}

        assert by_id["stub-abstains-on-contradiction"].metrics["abstained"] is True


class TestScoring:
    """Each check reads one property, and a run that produced nothing scores zero."""

    def test_a_forbidden_claim_fails(self):
        state = AgentState(final_report="The revenue was $4.2bn.")

        checks = harness.score_case(case(answer_excludes=[r"4\.2bn"]), state, None)

        assert [check["passed"] for check in checks] == [False]

    def test_a_required_abstain_that_did_not_fire_fails(self):
        state = AgentState(final_report="It was worth $1bn.")

        checks = harness.score_case(case(abstain="required"), state, None)

        assert [check["passed"] for check in checks] == [False]

    def test_a_forbidden_abstain_that_did_fire_fails(self):
        state = AgentState(
            final_report="I could not verify the evidence.",
            context={"verification": "abstained"},
        )

        checks = harness.score_case(case(abstain="forbidden"), state, None)

        assert [check["passed"] for check in checks] == [False]

    def test_an_ungraded_run_fails_validation_required(self):
        state = AgentState(final_report="A claim with no verdict behind it.")

        checks = harness.score_case(case(validation_required=True), state, None)

        assert [check["passed"] for check in checks] == [False]

    def test_the_page_check_needs_a_complete_document(self):
        fragment = AgentState(final_report='<section class="hero"><h1>Hi</h1></section>')
        page = AgentState(final_report="<!doctype html><html><body>x</body></html>")

        assert not harness.score_case(case(page=True), fragment, None)[0]["passed"]
        assert harness.score_case(case(page=True), page, None)[0]["passed"]

    def test_a_run_that_produced_no_state_fails_on_the_run(self):
        checks = harness.score_case(case(answer_min_words=1), None, "RuntimeError: boom")

        assert checks == [{"check": "run", "passed": False, "detail": "RuntimeError: boom"}]


class TestBaselineComparison:
    """A regression has to be a number, not an opinion."""

    def test_a_lower_score_is_a_regression(self):
        baseline = harness.report_payload([result("a", 1.0), result("b", 0.5)], "live", "m")

        comparison = harness.compare([result("a", 0.8), result("b", 0.5)], baseline)

        assert comparison["regressed"] == ["a"]
        assert comparison["rows"][0]["delta"] == -0.2
        assert comparison["overall_now"] < comparison["overall_then"]

    def test_an_unseen_case_is_reported_as_new(self):
        baseline = harness.report_payload([result("a", 1.0)], "live", "m")

        comparison = harness.compare([result("a", 1.0), result("c", 0.0)], baseline)

        assert comparison["regressed"] == []
        assert comparison["rows"][1]["delta"] is None

    def test_a_filtered_run_compares_only_the_cases_it_holds(self):
        baseline = harness.report_payload([result("a", 1.0), result("b", 0.0)], "live", "m")

        comparison = harness.compare([result("a", 1.0)], baseline)

        assert comparison["compared"] == 1
        assert comparison["overall_then"] == 1.0
        assert comparison["overall_now"] == 1.0


class TestCaseFile:
    """The case file must stay gradeable: known checks, known modes, unique ids."""

    def test_every_case_loads_and_the_three_modes_are_covered(self):
        cases = harness.load_cases()

        assert len(cases) >= 15
        assert {item.mode for item in cases} == set(harness.MODES)
        assert all(item.checks and item.question and item.why for item in cases)
