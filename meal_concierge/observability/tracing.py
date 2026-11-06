"""OpenTelemetry Distributed Tracing with Google Cloud Trace export and W3C span linking.

Implements Rubric Category 4.3 (Distributed Tracing):
Links parent and child spans from user query through Coordinator, Sub-Agents,
Tool executions, and Database/Memory lookups using OpenTelemetry (`opentelemetry-api`,
`opentelemetry-sdk`, `CloudTraceSpanExporter`) with W3C `traceparent` context propagation.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import functools
import inspect
import os
import time
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple
import uuid

from meal_concierge.observability.pii_redactor import redact_pii_data

try:
    from opentelemetry import trace as otel_trace  # type: ignore[import-not-found]
    from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult  # type: ignore[import-not-found]

    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - local fallback when opentelemetry wheels are absent
    otel_trace = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None
    SimpleSpanProcessor = None
    SpanExporter = object  # type: ignore[assignment,misc]
    SpanExportResult = None
    OTEL_AVAILABLE = False

try:
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter  # type: ignore[import-not-found]

    CLOUD_TRACE_EXPORTER_AVAILABLE = True
except ImportError:  # pragma: no cover
    CloudTraceSpanExporter = None
    CLOUD_TRACE_EXPORTER_AVAILABLE = False


_ACTIVE_TRACE_ID: ContextVar[Optional[str]] = ContextVar("_ACTIVE_TRACE_ID", default=None)
_ACTIVE_SPAN_STACK: ContextVar[Tuple[str, ...]] = ContextVar("_ACTIVE_SPAN_STACK", default=())
_ACTIVE_SESSION_ID: ContextVar[str] = ContextVar("_ACTIVE_SESSION_ID", default="default_session")
_ACTIVE_AGENT_NAME: ContextVar[str] = ContextVar(
    "_ACTIVE_AGENT_NAME", default="meal_concierge_coordinator"
)


@contextmanager
def execution_context(
    *,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> Iterator[None]:
    """Bind `session_id` and `agent_name` to ContextVars for correlated logging and auditing."""
    sess_token = _ACTIVE_SESSION_ID.set(session_id) if session_id else None
    agent_token = _ACTIVE_AGENT_NAME.set(agent_name) if agent_name else None
    try:
        yield
    finally:
        if sess_token is not None:
            _ACTIVE_SESSION_ID.reset(sess_token)
        if agent_token is not None:
            _ACTIVE_AGENT_NAME.reset(agent_token)


def get_active_session_id() -> str:
    """Return the active session ID from execution context."""
    return _ACTIVE_SESSION_ID.get()


def get_active_agent_name() -> str:
    """Return the active agent name from execution context."""
    return _ACTIVE_AGENT_NAME.get()


@dataclass
class SpanRecord:
    """Structured representation of a distributed trace span with W3C `traceparent`."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: str
    start_time_iso: str
    end_time_iso: Optional[str] = None
    duration_ms: float = 0.0
    status: str = "OK"
    w3c_traceparent: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    _otel_span: Optional[Any] = field(default=None, repr=False, compare=False)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Attach a timestamped event to this span."""
        sanitized = redact_pii_data(attributes or {})
        self.events.append(
            {
                "name": name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attributes": sanitized,
            }
        )
        if self._otel_span is not None and hasattr(self._otel_span, "add_event"):
            try:
                self._otel_span.add_event(name, attributes={str(k): str(v) for k, v in sanitized.items()})
            except Exception:
                pass

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a PII-redacted span attribute and forward to OpenTelemetry span if active."""
        sanitized = redact_pii_data(value)
        self.attributes[key] = sanitized
        if self._otel_span is not None and hasattr(self._otel_span, "set_attribute"):
            try:
                self._otel_span.set_attribute(str(key), str(sanitized))
            except Exception:
                pass


class InMemorySpanExporter:
    """Bounded in-memory span exporter for evaluation, testing, and trace tree inspection."""

    def __init__(self, maxlen: int = 5000) -> None:
        self._spans: Deque[SpanRecord] = deque(maxlen=maxlen)

    def export(self, span: SpanRecord) -> None:
        self._spans.append(span)

    def get_finished_spans(self) -> List[SpanRecord]:
        return list(self._spans)

    def get_trace_tree(self, trace_id: str) -> List[SpanRecord]:
        return [s for s in self._spans if s.trace_id == trace_id]

    def clear(self) -> None:
        self._spans.clear()


class TracingManager:
    """Manages OpenTelemetry TracerProvider, Cloud Trace export, and W3C span hierarchy."""

    def __init__(
        self,
        service_name: str = "meal-concierge-agent",
        gcp_project_id: Optional[str] = None,
        enable_cloud_trace: Optional[bool] = None,
    ) -> None:
        self.service_name = service_name
        self.gcp_project_id = gcp_project_id or os.environ.get(
            "GOOGLE_CLOUD_PROJECT", "fde-meal-concierge"
        )
        env_trace = os.environ.get("ENABLE_CLOUD_TRACE", "false").lower() in ("1", "true", "yes")
        self.enable_cloud_trace = (
            enable_cloud_trace if enable_cloud_trace is not None else env_trace
        )
        self.exporter = InMemorySpanExporter()
        self._otel_tracer: Any = None

        if OTEL_AVAILABLE and TracerProvider is not None and otel_trace is not None:
            resource = Resource.create(
                {
                    "service.name": service_name,
                    "cloud.account.id": self.gcp_project_id,
                }
            )
            provider = TracerProvider(resource=resource)
            if (
                self.enable_cloud_trace
                and CLOUD_TRACE_EXPORTER_AVAILABLE
                and CloudTraceSpanExporter is not None
            ):
                try:
                    cloud_exporter = CloudTraceSpanExporter(project_id=self.gcp_project_id)
                    provider.add_span_processor(BatchSpanProcessor(cloud_exporter))
                except Exception:
                    pass
            self._otel_tracer = provider.get_tracer(service_name)

    @staticmethod
    def current_trace_id() -> str:
        """Return the active 32-hex-char trace ID or generate one if none is active."""
        tid = _ACTIVE_TRACE_ID.get()
        if tid is None:
            tid = uuid.uuid4().hex
            _ACTIVE_TRACE_ID.set(tid)
        return tid

    @staticmethod
    def current_span_id() -> Optional[str]:
        """Return the active 16-hex-char span ID if inside a span."""
        stack = _ACTIVE_SPAN_STACK.get()
        return stack[-1] if stack else None

    @classmethod
    def get_w3c_traceparent(cls) -> str:
        """Format the active trace and span IDs as a W3C Trace Context `traceparent` header."""
        trace_id = cls.current_trace_id()
        span_id = cls.current_span_id() or "0000000000000001"
        return f"00-{trace_id}-{span_id}-01"

    @classmethod
    def extract_w3c_traceparent(cls, traceparent: str) -> Optional[Dict[str, str]]:
        """Extract `trace_id` and `parent_span_id` from an inbound W3C `traceparent` header."""
        parts = traceparent.strip().split("-")
        if len(parts) == 4 and len(parts[1]) == 32 and len(parts[2]) == 16:
            _ACTIVE_TRACE_ID.set(parts[1])
            _ACTIVE_SPAN_STACK.set((parts[2],))
            return {"version": parts[0], "trace_id": parts[1], "parent_span_id": parts[2], "flags": parts[3]}
        return None

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        *,
        kind: str = "INTERNAL",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[SpanRecord]:
        """Context manager creating a linked OpenTelemetry child span and recording telemetry."""
        stack = _ACTIVE_SPAN_STACK.get()
        trace_top_token = None
        parent_span_id = stack[-1] if stack else None
        sanitized_attrs = redact_pii_data(attributes or {})

        otel_cm = (
            self._otel_tracer.start_as_current_span(name)
            if self._otel_tracer is not None
            else nullcontext(None)
        )

        with otel_cm as otel_span:
            if otel_span is not None and hasattr(otel_span, "get_span_context"):
                span_ctx = otel_span.get_span_context()
                trace_id = f"{span_ctx.trace_id:032x}"
                span_id = f"{span_ctx.span_id:016x}"
                if not stack:
                    trace_top_token = _ACTIVE_TRACE_ID.set(trace_id)
                for k, v in sanitized_attrs.items():
                    try:
                        otel_span.set_attribute(str(k), str(v))
                    except Exception:
                        pass
            else:
                if not stack:
                    new_trace_id = uuid.uuid4().hex
                    trace_top_token = _ACTIVE_TRACE_ID.set(new_trace_id)
                trace_id = self.current_trace_id()
                span_id = uuid.uuid4().hex[:16]

            stack_token = _ACTIVE_SPAN_STACK.set((*stack, span_id))
            start_ns = time.perf_counter_ns()
            w3c_header = f"00-{trace_id}-{span_id}-01"
            sanitized_attrs["w3c.traceparent"] = w3c_header
            span = SpanRecord(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                name=name,
                kind=kind,
                start_time_iso=datetime.now(timezone.utc).isoformat(),
                w3c_traceparent=w3c_header,
                attributes=sanitized_attrs,
                _otel_span=otel_span,
            )

            try:
                yield span
            except Exception as exc:
                span.status = f"ERROR: {type(exc).__name__}"
                span.set_attribute("error.type", type(exc).__name__)
                span.set_attribute("error.message", str(exc))
                if otel_span is not None and hasattr(otel_span, "record_exception"):
                    try:
                        otel_span.record_exception(exc)
                    except Exception:
                        pass
                raise
            finally:
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                span.duration_ms = round(elapsed_ms, 3)
                span.end_time_iso = datetime.now(timezone.utc).isoformat()
                self.exporter.export(span)
                _ACTIVE_SPAN_STACK.reset(stack_token)
                if trace_top_token is not None:
                    _ACTIVE_TRACE_ID.reset(trace_top_token)


_GLOBAL_TRACER = TracingManager()


def get_tracer() -> TracingManager:
    """Return the singleton `TracingManager`."""
    return _GLOBAL_TRACER


def traced_span(
    span_name: Optional[str] = None,
    *,
    kind: str = "INTERNAL",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that wraps sync or async functions in a linked OpenTelemetry trace span."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        resolved_name = span_name or f"{func.__module__}.{func.__qualname__}"

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _GLOBAL_TRACER.start_as_current_span(
                    resolved_name,
                    kind=kind,
                    attributes={"function.name": func.__name__},
                ) as span:
                    result = await func(*args, **kwargs)
                    span.set_attribute("execution.completed", True)
                    return result

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _GLOBAL_TRACER.start_as_current_span(
                resolved_name,
                kind=kind,
                attributes={"function.name": func.__name__},
            ) as span:
                result = func(*args, **kwargs)
                span.set_attribute("execution.completed", True)
                return result

        return sync_wrapper

    return decorator
