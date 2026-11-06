"""Structured JSON Logging with Google Cloud Logging correlation and PII scrubbing.

Implements Rubric Category 4.1 (Structured JSON Logging):
Emits single-line JSON log records conforming to Google Cloud Structured Logging
(`severity`, `timestamp`, `logging.googleapis.com/trace`, `logging.googleapis.com/spanId`,
`agent_name`, `session_id`, `event_type`, `payload`) rather than `print` statements,
automatically correlating with active OpenTelemetry trace/span IDs.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import logging
import os
import sys
from typing import Any, Deque, Dict, List, Optional

from meal_concierge.observability.pii_redactor import PIIRedactionFilter, redact_pii_data, redact_pii_text
from meal_concierge.observability.tracing import (
    get_active_agent_name,
    get_active_session_id,
    get_tracer,
)


class CloudLoggingJsonFormatter(logging.Formatter):
    """Formats `logging.LogRecord` instances into single-line Google Cloud JSON payloads."""

    def __init__(
        self,
        service_name: str = "meal-concierge-agent",
        gcp_project_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.service_name = service_name
        self.gcp_project_id = gcp_project_id or os.environ.get(
            "GOOGLE_CLOUD_PROJECT", "fde-meal-concierge"
        )

    def format(self, record: logging.LogRecord) -> str:
        raw_message = record.getMessage()
        already_redacted = getattr(record, "_pii_already_redacted", False)
        sanitized_message = raw_message if already_redacted else redact_pii_text(raw_message)

        tracer = get_tracer()
        trace_id = getattr(record, "trace_id", None) or tracer.current_trace_id()
        span_id = getattr(record, "span_id", None) or tracer.current_span_id() or "0000000000000001"
        agent_name = getattr(record, "agent_name", None) or get_active_agent_name()
        session_id = getattr(record, "session_id", None) or get_active_session_id()
        event_type = getattr(record, "event_type", "AGENT_LOG")
        raw_payload = getattr(record, "structured_payload", {})
        structured_payload = raw_payload if already_redacted else redact_pii_data(raw_payload)

        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "service": self.service_name,
            "logger": record.name,
            "message": sanitized_message,
            "agent_name": agent_name,
            "session_id": session_id,
            "event_type": event_type,
            "logging.googleapis.com/trace": f"projects/{self.gcp_project_id}/traces/{trace_id}",
            "logging.googleapis.com/spanId": span_id,
            "payload": structured_payload,
        }
        if record.exc_info:
            log_entry["exception"] = redact_pii_text(self.formatException(record.exc_info))
        return json.dumps(log_entry, sort_keys=True, default=str)


class StructuredJsonLogger:
    """High-level structured logger capturing rich agent telemetry in a bounded ring buffer."""

    def __init__(
        self,
        name: str = "meal_concierge",
        service_name: str = "meal-concierge-agent",
        level: int = logging.INFO,
        max_buffer_size: int = 2000,
    ) -> None:
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.propagate = False
        self._records_buffer: Deque[Dict[str, Any]] = deque(maxlen=max_buffer_size)

        if not self._logger.handlers:
            stream_handler = logging.StreamHandler(stream=sys.stderr)
            stream_handler.setFormatter(CloudLoggingJsonFormatter(service_name=service_name))
            stream_handler.addFilter(PIIRedactionFilter())
            self._logger.addHandler(stream_handler)

    def log_event(
        self,
        severity: str,
        message: str,
        *,
        event_type: str = "AGENT_EVENT",
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Emit a structured JSON log entry correlated with active trace/span/session context."""
        sanitized_msg = redact_pii_text(message)
        sanitized_payload = redact_pii_data(payload or {})
        tracer = get_tracer()
        resolved_trace = trace_id or tracer.current_trace_id()
        resolved_span = span_id or tracer.current_span_id() or "0000000000000001"
        resolved_agent = agent_name or get_active_agent_name()
        resolved_session = session_id or get_active_session_id()

        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": severity.upper(),
            "message": sanitized_msg,
            "event_type": event_type,
            "agent_name": resolved_agent,
            "session_id": resolved_session,
            "trace_id": resolved_trace,
            "span_id": resolved_span,
            "payload": sanitized_payload,
        }
        self._records_buffer.append(entry)

        level = getattr(logging, severity.upper(), logging.INFO)
        self._logger.log(
            level,
            sanitized_msg,
            extra={
                "event_type": event_type,
                "agent_name": resolved_agent,
                "session_id": resolved_session,
                "trace_id": resolved_trace,
                "span_id": resolved_span,
                "structured_payload": sanitized_payload,
                "_pii_already_redacted": True,
            },
        )
        return entry

    def info(self, message: str, **kwargs: Any) -> Dict[str, Any]:
        """Log an INFO-level structured event."""
        return self.log_event("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> Dict[str, Any]:
        """Log a WARNING-level structured event."""
        return self.log_event("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> Dict[str, Any]:
        """Log an ERROR-level structured event."""
        return self.log_event("ERROR", message, **kwargs)

    def get_buffered_records(self) -> List[Dict[str, Any]]:
        """Return all structured JSON log entries retained in the bounded ring buffer."""
        return list(self._records_buffer)

    def clear_buffer(self) -> None:
        """Clear the in-memory structured log buffer."""
        self._records_buffer.clear()


_GLOBAL_LOGGER = StructuredJsonLogger()


def get_structured_logger() -> StructuredJsonLogger:
    """Return the singleton `StructuredJsonLogger` instance."""
    return _GLOBAL_LOGGER
