"""python -m evals.run - score the answer-quality cases.

Without ``--live`` this runs the scripted suite: no key, no gateway, no network,
and it fails if the harness stops discriminating between a grounded answer, a
fabricated one, and a run that should have abstained. With ``--live`` it runs
the cases in ``evals/cases.json`` against the KeyPool gateway.

Exit codes: 0 every case passed, 1 a case failed, 2 a case regressed against the
baseline, 3 the harness or its setup is broken.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from evals.env import isolate

KEYPOOL_ENV = Path.home() / ".capy" / "work" / "keypool.env"
"""Where this machine keeps the gateway credentials, for runtime use only."""

EXIT_OK = 0
EXIT_CASE_FAILED = 1
EXIT_REGRESSION = 2
EXIT_SETUP = 3

HEADERS = ("case", "mode", "score", "checks", "secs", "calls", "tokens", "verdicts")
"""The columns of the result table, in order."""


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run",
        description="Score the quality of the system's answers.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="run evals/cases.json against the gateway (needs its credentials)",
    )
    parser.add_argument(
        "--stub", action="store_true", help="run the scripted harness suite (the default)"
    )
    parser.add_argument("--case", action="append", metavar="ID", help="run only these cases")
    parser.add_argument("--mode", action="append", metavar="MODE", help="pipeline, team or build")
    parser.add_argument("--baseline", metavar="PATH", help="compare against this saved run")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="save this run as evals/baseline.json",
    )
    parser.add_argument("--json", metavar="PATH", help="write the run record here")
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        metavar="SECONDS",
        help="wall-clock cap on one case (default 300)",
    )
    parser.add_argument("--list", action="store_true", help="list the cases and exit")
    parser.add_argument(
        "--keypool-env",
        metavar="PATH",
        default=str(KEYPOOL_ENV),
        help="file holding the gateway credentials",
    )
    return parser.parse_args(argv)


def load_gateway_credentials(path: Path) -> str:
    """Point this process at the KeyPool gateway, without printing any value."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.strip().partition("=")
            if not sep or not name or name.startswith("#"):
                continue
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))
    os.environ.setdefault("LLM_PROVIDER", "keypool")
    os.environ.setdefault("KEYPOOL_BASE_URL", "http://127.0.0.1:8787")
    os.environ.setdefault("KEYPOOL_MODEL", "pooled-model")
    os.environ.setdefault("ENABLE_FREE_FALLBACK", "false")
    os.environ.setdefault("THINKING_LEVEL", "off")
    return str(path)


def _line(cells: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)).rstrip()


def print_table(results: list[Any]) -> None:
    """One row per case, with what it cost."""
    rows: list[tuple[str, ...]] = []
    for result in results:
        verdicts = result.metrics["verdicts"]
        graded = f"{verdicts['supported']}/{verdicts['unsupported']}/{verdicts['contradicted']}"
        earned = sum(1 for check in result.checks if check["passed"])
        rows.append(
            (
                result.id,
                result.mode,
                f"{result.score:.2f}",
                f"{earned}/{len(result.checks)}",
                f"{result.metrics['seconds']:.1f}",
                str(result.metrics["model_calls"]),
                str(result.metrics["tokens"] or "-"),
                graded if any(verdicts.values()) else "-",
            )
        )
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(HEADERS)
    ]
    print(_line(HEADERS, widths))
    print(_line(tuple("-" * width for width in widths), widths))
    for row in rows:
        print(_line(row, widths))
    print("verdicts: supported/unsupported/contradicted for the claims this run graded")


def print_failures(results: list[Any]) -> None:
    """The check that failed on each case, and what it saw."""
    failed = [result for result in results if not result.passed]
    if not failed:
        return
    print("\nFAILED CHECKS")
    for result in failed:
        print(f"  {result.id}")
        for check in result.checks:
            if not check["passed"]:
                print(f"    {check['check']}: {check['detail']}")


def print_summary(summary: dict[str, Any]) -> None:
    """The run's score and cost in one block."""
    verdicts = summary["verdicts"]
    print(
        f"\nScore {summary['score']:.2f}  ({summary['passed']}/{summary['cases']} cases passed)"
        f"  {summary['seconds']:.1f}s  {summary['model_calls']} model calls"
        f"  {summary['tokens']} tokens"
    )
    print(
        f"Verdicts: {verdicts['supported']} supported, {verdicts['unsupported']} unsupported,"
        f" {verdicts['contradicted']} contradicted"
    )
    if summary["failed"]:
        print(f"Failing: {', '.join(summary['failed'])}")


def print_comparison(comparison: dict[str, Any], baseline: dict[str, Any], path: Path) -> None:
    """Per-case deltas against a saved run, worst first."""
    print(f"\nBASELINE {path} measured {baseline.get('generated_at', 'unknown')}")
    rows = sorted(
        comparison["rows"],
        key=lambda row: (row["delta"] is None, row["delta"] if row["delta"] is not None else 0),
    )
    headers = ("case", "then", "now", "delta")
    cells = [
        (
            row["id"],
            "-" if row["then"] is None else f"{row['then']:.2f}",
            f"{row['now']:.2f}",
            "new" if row["delta"] is None else f"{row['delta']:+.2f}",
        )
        for row in rows
    ]
    widths = [max(len(headers[i]), *(len(cell[i]) for cell in cells)) for i in range(4)]
    print(_line(headers, widths))
    print(_line(tuple("-" * width for width in widths), widths))
    for cell in cells:
        print(_line(cell, widths))
    if comparison["overall_then"] is None:
        print("overall: no case in this run appears in the baseline")
    else:
        change = comparison["overall_now"] - comparison["overall_then"]
        print(
            f"overall over {comparison['compared']} compared case(s):"
            f" {comparison['overall_then']:.2f} -> {comparison['overall_now']:.2f}"
            f"  ({change:+.2f})"
        )
    if comparison["regressed"]:
        print(f"REGRESSED: {', '.join(comparison['regressed'])}")


async def run_stub() -> int:
    """Run the scripted suite and check the harness itself."""
    from evals import harness, stub

    results = await stub.run_stub_suite()
    print("Mode: stub (scripted model, no credentials, no network)\n")
    print_table(results)
    print_failures(results)
    print_summary(harness.summarize(results))

    problems = stub.selftest(results)
    if problems:
        print("\nHARNESS FAILED")
        for problem in problems:
            print(f"  {problem}")
        return EXIT_SETUP
    print(
        "\nHarness selftest: a grounded answer passes, a fabricated one fails its exclusion,"
        " and a fully contradicted run abstains."
    )
    print(f"{len(harness.load_cases())} measurement cases in evals/cases.json need --live.")
    return EXIT_OK


class SetupError(RuntimeError):
    """The run cannot start, and the message says why."""


def _selected_cases(args: argparse.Namespace) -> list[Any]:
    """The cases this run asked for, filtered by id and mode."""
    from evals import harness

    cases = harness.load_cases()
    if args.case:
        wanted = set(args.case)
        missing = sorted(wanted - {case.id for case in cases})
        if missing:
            raise SetupError(f"no such case: {', '.join(missing)}")
        cases = [case for case in cases if case.id in wanted]
    if args.mode:
        modes = set(args.mode)
        unknown = sorted(modes - set(harness.MODES))
        if unknown:
            raise SetupError(f"no such mode: {', '.join(unknown)}")
        cases = [case for case in cases if case.mode in modes]
    if not cases:
        raise SetupError("no cases selected")
    return cases


async def _connect() -> tuple[Any, Any]:
    """The model and provider to run against, or a setup error."""
    from src.config import get_llm, router

    try:
        llm = get_llm()
    except RuntimeError as exc:
        raise SetupError(f"no model configured: {exc}") from exc
    provider = router.providers[0] if router.providers else None
    if provider is None or not await router.check(provider):
        raise SetupError(f"the gateway at {os.environ.get('KEYPOOL_BASE_URL')} is not answering")
    return llm, provider


async def run_live(args: argparse.Namespace) -> int:
    """Run the case file against the gateway and compare with a saved baseline."""
    from evals import harness

    try:
        cases = _selected_cases(args)
        llm, provider = await _connect()
    except (OSError, ValueError, KeyError) as exc:
        print(f"case file is unreadable: {exc}")
        return EXIT_SETUP
    except SetupError as exc:
        print(str(exc))
        return EXIT_SETUP

    print(
        f"Mode: live ({provider.name}, {provider.model})"
        f"  {len(cases)} cases  timeout {args.timeout:.0f}s\n"
    )
    results = await harness.evaluate(
        cases, lambda meter: harness.live_model(llm, meter), timeout=args.timeout
    )
    payload = harness.report_payload(results, "live", provider.model)
    print_table(results)
    print_failures(results)
    summary = payload["summary"]
    print_summary(summary)

    record = Path(args.json) if args.json else harness.LAST_FILE
    harness.write_payload(record, payload)
    print(f"\nRun record: {record}")
    if args.save_baseline:
        harness.write_payload(harness.BASELINE_FILE, payload)
        print(f"Baseline:   {harness.BASELINE_FILE}")

    return _finish(results, summary, args)


def _finish(results: list[Any], summary: dict[str, Any], args: argparse.Namespace) -> int:
    """Exit on the score, then on any regression against the baseline."""
    from evals import harness

    code = EXIT_OK if summary["failed"] == [] else EXIT_CASE_FAILED
    compared = Path(args.baseline) if args.baseline else harness.BASELINE_FILE
    if not compared.exists():
        return code
    baseline = json.loads(compared.read_text(encoding="utf-8"))
    comparison = harness.compare(results, baseline)
    print_comparison(comparison, baseline, compared)
    return EXIT_REGRESSION if comparison["regressed"] else code


def main(argv: list[str] | None = None) -> int:
    """Run the requested suite."""
    isolate()
    args = _parse_args(argv)
    if args.live and not os.environ.get("KEYPOOL_API_KEY"):
        load_gateway_credentials(Path(args.keypool_env))
    if args.live and not os.environ.get("KEYPOOL_API_KEY"):
        print(
            f"no gateway credential: put KEYPOOL_API_KEY in {args.keypool_env} or the environment"
        )
        return EXIT_SETUP
    if args.list:
        return asyncio.run(_list_cases())
    return asyncio.run(run_live(args) if args.live else run_stub())


async def _list_cases() -> int:
    """Print the case file without running anything."""
    from evals import harness

    try:
        cases = harness.load_cases()
    except (OSError, ValueError, KeyError) as exc:
        print(f"case file is unreadable: {exc}")
        return EXIT_SETUP
    headers = ("id", "mode", "checks", "why")
    cells = [(case.id, case.mode, str(len(case.checks)), case.why) for case in cases]
    widths = [max(len(headers[i]), *(len(cell[i]) for cell in cells)) for i in range(4)]
    print(_line(headers, widths))
    for cell in cells:
        print(_line(cell, widths))
    print(f"\n{len(cases)} cases")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
