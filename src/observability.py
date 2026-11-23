"""Observability setup: OpenTelemetry, LangSmith, and custom tracing."""

import logging
import os
from collections.abc import Awaitable, Callable
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
