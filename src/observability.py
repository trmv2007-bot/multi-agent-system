"""Observability setup: OpenTelemetry, LangSmith, custom tracing, and run cost.

Spans and metrics answer "what happened"; :class:`RunMeter` answers "what did
this run cost" - how many model calls, how many tokens, how many seconds, which
provider served each one, and whether any of them came back as JSON that needed
repairing. A run is one turn: the dashboard starts one per message, the CLI one
per query, and :func:`run_summary` reports it to anything that wants the numbers
without reading a trace.
"""

import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from types import TracebackType
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode

from src.config import settings


def _resource() -> Resource:
    """Describe this service to the collector."""
    return Resource.create(
        {
            SERVICE_NAME: settings.otel_service_name,
            SERVICE_VERSION: "0.1.0",
            "deployment.environment": "development",
        }
    )


def _debug() -> bool:
    """Whether span and metric payloads should also be printed locally."""
    return settings.log_level.upper() == "DEBUG"


def _tracer_provider(resource: Resource) -> TracerProvider:
    """Build the tracer provider, exporting only when OTel is enabled."""
    provider = TracerProvider(resource=resource)
    if not settings.otel_enabled:
        return provider

    try:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True))
        )
    except Exception as e:
        logging.warning(f"Failed to setup OTLP trace exporter: {e}")

    if _debug():
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    return provider


def _meter_provider(resource: Resource) -> MeterProvider:
    """Build the meter provider, exporting only when OTel is enabled."""
    readers = []
    if settings.otel_enabled:
        try:
            readers.append(
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=settings.otel_endpoint, insecure=True)
                )
            )
        except Exception as e:
            logging.warning(f"Failed to setup OTLP metric exporter: {e}")

        if _debug():
            readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))

    return MeterProvider(resource=resource, metric_readers=readers)


def _instrument() -> None:
    """Attach the auto-instrumentors available in this install."""
    for instrumentor, label in (
        (LangchainInstrumentor(), "LangChain"),
        (HTTPXClientInstrumentor(), "HTTPX"),
        (RequestsInstrumentor(), "Requests"),
    ):
        try:
            instrumentor.instrument()
        except Exception as e:
            logging.warning(f"Failed to instrument {label}: {e}")


def setup_observability() -> trace.Tracer:
    """
    Setup the observability stack.

    - OpenTelemetry tracing and metrics (exported only when OTEL_ENABLED=true)
    - LangSmith tracing when a LangSmith key is configured
    - Auto-instrumentation for LangChain, httpx, and requests
    """
    resource = _resource()
    trace.set_tracer_provider(_tracer_provider(resource))
    metrics.set_meter_provider(_meter_provider(resource))
    _instrument()

    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    return trace.get_tracer(__name__)


class TraceContext:
    """Context manager for custom spans with attributes."""

    def __init__(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.name = name
        self.attributes = attributes or {}
        self.span: trace.Span | None = None

    def __enter__(self) -> trace.Span:
        self.span = trace.get_tracer(__name__).start_span(self.name)
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        return self.span

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        span = self.span
        if span is None:
            return
        if exc_val is not None:
            span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            span.record_exception(exc_val)
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()


def record_metric(name: str, value: float, attributes: dict[str, Any] | None = None) -> None:
    """Record a metric value."""
    metrics.get_meter(__name__).create_counter(name).add(value, attributes or {})


def record_histogram(name: str, value: float, attributes: dict[str, Any] | None = None) -> None:
    """Record a histogram value."""
    metrics.get_meter(__name__).create_histogram(name).record(value, attributes or {})


def structured_output_stats() -> dict[str, int]:
    """How model JSON replies have gone since this process started.

    Repairs and retries climbing means the provider pool is mangling output;
    ``empty`` climbing with ``calls`` means it is burning the budget on reasoning
    again. Either way the run stayed up, so nothing else would have said so.
    """
    from src.structured import STRUCTURED

    return STRUCTURED.snapshot()


def trace_agent(agent_name: str) -> Callable[..., Any]:
    """Decorator to trace agent execution."""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with TraceContext(f"agent.{agent_name}", {"agent.name": agent_name}) as span:
                try:
                    result = await func(*args, **kwargs)
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    raise
                span.set_attribute("success", True)
                return result

        return wrapper

    return decorator


def trace_tool(tool_name: str) -> Callable[..., Any]:
    """Decorator to trace tool execution."""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with TraceContext(f"tool.{tool_name}", {"tool.name": tool_name}) as span:
                try:
                    result = await func(*args, **kwargs)
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    raise
                span.set_attribute("success", True)
                return result

        return wrapper

    return decorator


# ── per-run accounting ───────────────────────────────────────────────
@dataclass(frozen=True)
class ModelCall:
    """One model call, as a run's books record it."""

    agent: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class RunMeter:
    """What one run cost, accumulated while it happens.

    A run is one turn. Parallel agents report into the same meter on one event
    loop, so recording is a lock and an append; nothing here is shared between
    runs, and the JSON-repair counters are reported as the delta a run added to
    the process-wide totals in :mod:`src.structured`.
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.started = time.monotonic()
        self.stopped: float | None = None
        self._lock = threading.Lock()
        self._calls: list[ModelCall] = []
        self._structured_from = structured_output_stats()
        self._trace: TraceContext | None = None

    def start(self) -> "RunMeter":
        """Open the run's span, so the cost also lands in the trace."""
        self._trace = TraceContext("run", {"run.label": self.label or "run"})
        self._trace.__enter__()
        return self

    def record(self, agent: str, response: Any, seconds: float, provider: str = "") -> None:
        """Book one finished model call and the tokens its provider reported."""
        prompt, completion = _tokens_of(response)
        call = ModelCall(
            agent=agent,
            provider=provider,
            model=_model_of(response),
            prompt_tokens=prompt,
            completion_tokens=completion,
            seconds=seconds,
        )
        with self._lock:
            self._calls.append(call)

    def stop(self) -> None:
        """Close the books, and the run's span with them."""
        if self.stopped is None:
            self.stopped = time.monotonic()
        if self._trace is None:
            return
        span = self._trace.span
        if span is not None:
            summary = self.summary()
            span.set_attribute("run.calls", summary["calls"])
            span.set_attribute("run.tokens", summary["tokens"])
            span.set_attribute("run.seconds", summary["seconds"])
            span.set_attribute("run.slowest_agent", summary["slowest_agent"])
            span.set_attribute("run.json_repaired", summary["structured"]["repaired"])
        self._trace.__exit__(None, None, None)
        self._trace = None

    def summary(self) -> dict[str, Any]:
        """The run so far, in the shape :func:`format_run_summary` prints."""
        with self._lock:
            calls = list(self._calls)
        ended = self.stopped if self.stopped is not None else time.monotonic()

        agents: dict[str, dict[str, Any]] = {}
        models: dict[tuple[str, str], dict[str, Any]] = {}
        for call in calls:
            tokens = call.prompt_tokens + call.completion_tokens
            name = call.agent or "unattributed"
            agent = agents.setdefault(
                name, {"name": name, "calls": 0, "tokens": 0, "seconds": 0.0}
            )
            served = models.setdefault(
                (call.provider, call.model),
                {"provider": call.provider, "model": call.model, "calls": 0, "tokens": 0},
            )
            agent["calls"] += 1
            agent["tokens"] += tokens
            agent["seconds"] += call.seconds
            served["calls"] += 1
            served["tokens"] += tokens

        for agent in agents.values():
            agent["seconds"] = round(agent["seconds"], 2)
        ordered_agents = sorted(
            agents.values(), key=lambda row: (-row["seconds"], -row["tokens"], row["name"])
        )

        return {
            "label": self.label,
            "calls": len(calls),
            "prompt_tokens": sum(call.prompt_tokens for call in calls),
            "completion_tokens": sum(call.completion_tokens for call in calls),
            "tokens": sum(call.prompt_tokens + call.completion_tokens for call in calls),
            "seconds": round(ended - self.started, 2),
            "model_seconds": round(sum(call.seconds for call in calls), 2),
            "slowest_agent": ordered_agents[0]["name"] if ordered_agents else "",
            "agents": ordered_agents,
            "models": sorted(models.values(), key=lambda row: (-row["calls"], row["provider"])),
            "structured": _structured_since(self._structured_from),
        }


_RUN: ContextVar[RunMeter | None] = ContextVar("run_meter", default=None)


def start_run(label: str = "") -> RunMeter:
    """Begin accounting one run, and make it the one model calls report into.

    The meter lives in a context variable, so a task that starts a run - a
    dashboard turn - keeps its own books while other turns run beside it.
    """
    meter = RunMeter(label).start()
    _RUN.set(meter)
    return meter


def current_run() -> RunMeter | None:
    """The run being accounted right now, or ``None`` outside a run."""
    return _RUN.get()


def end_run(meter: RunMeter | None = None) -> dict[str, Any]:
    """Close a run's books - the current one unless one is named - and report them."""
    target = meter or _RUN.get()
    if target is None:
        return RunMeter().summary()
    target.stop()
    summary = target.summary()
    if _RUN.get() is target:
        _RUN.set(None)
    return summary


def run_summary() -> dict[str, Any]:
    """What the current run has cost so far, or zeros when nothing is running.

    The one function a caller needs: the CLI prints it, an eval or a test can
    assert on it, and the dashboard renders the same numbers for a person.
    """
    meter = _RUN.get()
    return RunMeter().summary() if meter is None else meter.summary()


def record_model_call(agent: str, response: Any, seconds: float, provider: str = "") -> None:
    """Book one finished model call against the current run, when there is one."""
    meter = _RUN.get()
    if meter is not None:
        meter.record(agent, response, seconds, provider)


def format_run_summary(summary: dict[str, Any]) -> str:
    """One line saying what a run cost, for a terminal or a log."""
    structured = summary["structured"]
    parts = [
        f"{summary['calls']} model call{_plural(summary['calls'])}",
        f"{_compact(summary['tokens'])} tokens "
        f"({_compact(summary['prompt_tokens'])} in, {_compact(summary['completion_tokens'])} out)",
        f"{summary['seconds']:.1f}s",
    ]
    if summary["slowest_agent"]:
        parts.append(f"slowest {summary['slowest_agent']}")
    if structured["repaired"]:
        parts.append(f"{structured['repaired']} JSON repair{_plural(structured['repaired'])}")
    if structured["failed"]:
        parts.append(f"{structured['failed']} JSON failure{_plural(structured['failed'])}")
    return " · ".join(parts)


def _structured_since(baseline: dict[str, int]) -> dict[str, int]:
    """The JSON counters this run added to the process-wide totals."""
    current = structured_output_stats()
    return {field: max(0, count - baseline.get(field, 0)) for field, count in current.items()}


def _tokens_of(response: Any) -> tuple[int, int]:
    """Prompt and completion tokens a reply reports, or zeros when it reports none."""
    usage = getattr(response, "usage_metadata", None) or {}
    if isinstance(usage, dict) and usage:
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    metadata = getattr(response, "response_metadata", None) or {}
    tokens = metadata.get("token_usage") or metadata.get("usage") or {}
    if isinstance(tokens, dict):
        return int(tokens.get("prompt_tokens") or 0), int(tokens.get("completion_tokens") or 0)
    return 0, 0


def _model_of(response: Any) -> str:
    """The model id a reply says served it, when the provider says."""
    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("model_name") or metadata.get("model") or "")


def _compact(tokens: int) -> str:
    """A token count at the width a reader wants: 18342 -> 18k."""
    if tokens < 1000:
        return str(tokens)
    if tokens < 10_000:
        return f"{tokens / 1000:.1f}k"
    return f"{tokens / 1000:.0f}k"


def _plural(count: int) -> str:
    """The plural suffix a count takes, which one of does not."""
    return "" if count == 1 else "s"
