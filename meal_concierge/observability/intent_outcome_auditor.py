"""Intent vs. Outcome Auditor for ADK Agents and Tool Calls.

Implements Rubric Category 4.2 (Intent vs. Outcome Capture):
Explicitly records both the agent's *intended* action before execution
(`record_intent`) and the *actual* outcome after execution (`record_outcome`),
computing quantitative intent-vs-outcome deltas and preventing duplicate callback logs.
"""

from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional
import uuid

from meal_concierge.observability.pii_redactor import redact_pii_data
from meal_concierge.observability.structured_logger import StructuredJsonLogger, get_structured_logger
from meal_concierge.observability.tracing import (
    get_active_agent_name,
    get_active_session_id,
    get_tracer,
)


_ACTIVE_TOOL_INTENT_ID: ContextVar[Optional[str]] = ContextVar("_ACTIVE_TOOL_INTENT_ID", default=None)


@dataclass
class IntentRecord:
    """Captured pre-execution intent before an agent step or tool invocation runs."""

    intent_id: str
    timestamp_iso: str
    trace_id: str
    span_id: str
    session_id: str
    agent_name: str
    target_action: str
    rationale: str
    expected_constraints: Dict[str, Any]
    sanitized_inputs: Dict[str, Any]
    started_perf_ns: int = field(repr=False, default=0)


@dataclass
class OutcomeRecord:
    """Captured post-execution outcome linked to its originating `IntentRecord`."""

    intent_id: str
    timestamp_iso: str
    trace_id: str
    span_id: str
    session_id: str
    agent_name: str
    target_action: str
    status: str
    latency_ms: float
    intent_satisfied: bool
    intent_vs_outcome_delta: Dict[str, Any]
    sanitized_outputs: Dict[str, Any]


class IntentOutcomeAuditor:
    """Captures and correlates pre-execution intents with post-execution outcomes."""

    def __init__(
        self,
        logger: Optional[StructuredJsonLogger] = None,
        max_records: int = 2000,
    ) -> None:
        self.logger = logger or get_structured_logger()
        self.max_records = max_records
        self._intents: "OrderedDict[str, IntentRecord]" = OrderedDict()
        self._outcomes: "OrderedDict[str, OutcomeRecord]" = OrderedDict()

    @staticmethod
    def _compute_quantitative_delta(
        expected_constraints: Dict[str, Any],
        outputs: Dict[str, Any],
        status: str,
        intent_satisfied: bool,
    ) -> Dict[str, Any]:
        """Compute quantitative diffs between intended constraints and actual execution outputs."""
        delta: Dict[str, Any] = {
            "expected_status": "success",
            "actual_status": status,
            "intent_satisfied": intent_satisfied,
            "constraint_evaluations": {},
        }
        if "min_calories_kcal" in expected_constraints and "daily_calorie_target_kcal" in outputs:
            actual_cal = outputs["daily_calorie_target_kcal"]
            min_cal = expected_constraints["min_calories_kcal"]
            delta["constraint_evaluations"]["calorie_floor_margin_kcal"] = actual_cal - min_cal
        if "weekly_budget_usd" in expected_constraints and "net_estimated_cost_usd" in outputs:
            budget = float(expected_constraints["weekly_budget_usd"] or 0.0)
            net_cost = float(outputs["net_estimated_cost_usd"] or 0.0)
            delta["constraint_evaluations"]["budget_headroom_usd"] = round(budget - net_cost, 2)
        if "excluded_allergens" in expected_constraints:
            enforced = outputs.get("excluded_allergens_enforced", expected_constraints["excluded_allergens"])
            delta["constraint_evaluations"]["allergens_filtered"] = enforced
        return delta

    def record_intent(
        self,
        *,
        agent_name: Optional[str] = None,
        target_action: str,
        rationale: str,
        inputs: Dict[str, Any],
        expected_constraints: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> IntentRecord:
        """Log and persist the agent's intended action BEFORE execution (deduplicated per call)."""
        existing_id = _ACTIVE_TOOL_INTENT_ID.get()
        if existing_id and existing_id in self._intents:
            existing = self._intents[existing_id]
            if existing.target_action in (target_action, f"tool:{target_action}"):
                return existing

        tracer = get_tracer()
        trace_id = tracer.current_trace_id()
        span_id = tracer.current_span_id() or "0000000000000001"
        resolved_agent = get_active_agent_name() if agent_name is None else agent_name
        resolved_session = session_id or get_active_session_id()
        intent_id = f"intent-{uuid.uuid4().hex[:12]}"

        sanitized_inputs = redact_pii_data(inputs)
        sanitized_constraints = redact_pii_data(expected_constraints or {})

        record = IntentRecord(
            intent_id=intent_id,
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            trace_id=trace_id,
            span_id=span_id,
            session_id=resolved_session,
            agent_name=resolved_agent,
            target_action=target_action,
            rationale=rationale,
            expected_constraints=sanitized_constraints,
            sanitized_inputs=sanitized_inputs,
            started_perf_ns=time.perf_counter_ns(),
        )
        self._intents[intent_id] = record
        while len(self._intents) > self.max_records:
            self._intents.popitem(last=False)

        self.logger.info(
            f"[INTENT] Agent '{resolved_agent}' intends to execute '{target_action}': {rationale}",
            event_type="AGENT_INTENT_CAPTURED",
            agent_name=resolved_agent,
            session_id=resolved_session,
            trace_id=trace_id,
            span_id=span_id,
            payload={
                "intent_id": intent_id,
                "target_action": target_action,
                "rationale": rationale,
                "expected_constraints": sanitized_constraints,
                "inputs": sanitized_inputs,
            },
        )
        return record

    def record_outcome(
        self,
        intent_id: str,
        *,
        status: str,
        outputs: Dict[str, Any],
        intent_satisfied: bool = True,
        delta_notes: Optional[Dict[str, Any]] = None,
    ) -> OutcomeRecord:
        """Log and persist the actual outcome AFTER execution, linked to `intent_id`."""
        if intent_id in self._outcomes:
            return self._outcomes[intent_id]

        intent = self._intents.get(intent_id)
        now_ns = time.perf_counter_ns()
        if intent is not None:
            latency_ms = round((now_ns - intent.started_perf_ns) / 1_000_000.0, 3)
            trace_id = intent.trace_id
            span_id = intent.span_id
            session_id = intent.session_id
            agent_name = intent.agent_name
            target_action = intent.target_action
            expected_constraints = intent.expected_constraints
        else:
            latency_ms = 0.0
            trace_id = get_tracer().current_trace_id()
            span_id = get_tracer().current_span_id() or "0000000000000001"
            session_id = get_active_session_id()
            agent_name = get_active_agent_name()
            target_action = "unknown_action"
            expected_constraints = {}

        sanitized_outputs = redact_pii_data(outputs)
        computed_delta = redact_pii_data(
            delta_notes
            or self._compute_quantitative_delta(
                expected_constraints,
                sanitized_outputs,
                status,
                intent_satisfied,
            )
        )

        outcome = OutcomeRecord(
            intent_id=intent_id,
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            trace_id=trace_id,
            span_id=span_id,
            session_id=session_id,
            agent_name=agent_name,
            target_action=target_action,
            status=status,
            latency_ms=latency_ms,
            intent_satisfied=intent_satisfied,
            intent_vs_outcome_delta=computed_delta,
            sanitized_outputs=sanitized_outputs,
        )
        self._outcomes[intent_id] = outcome
        while len(self._outcomes) > self.max_records:
            self._outcomes.popitem(last=False)

        severity = "INFO" if intent_satisfied and status.upper() != "ERROR" else "WARNING"
        self.logger.log_event(
            severity,
            f"[OUTCOME] Action '{target_action}' completed with status='{status}' (intent_satisfied={intent_satisfied}, latency={latency_ms}ms)",
            event_type="AGENT_OUTCOME_CAPTURED",
            agent_name=agent_name,
            session_id=session_id,
            trace_id=trace_id,
            span_id=span_id,
            payload={
                "intent_id": intent_id,
                "target_action": target_action,
                "status": status,
                "latency_ms": latency_ms,
                "intent_satisfied": intent_satisfied,
                "intent_vs_outcome_delta": computed_delta,
                "outputs": sanitized_outputs,
            },
        )
        return outcome

    def adk_before_tool_callback(
        self,
        tool: Any,
        args: Dict[str, Any],
        tool_context: Any,
    ) -> Optional[Dict[str, Any]]:
        """ADK `before_tool_callback` hook recording pre-execution tool intent without duplication."""
        tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
        agent_name = getattr(tool_context, "agent_name", None) or get_active_agent_name()
        session_id = getattr(tool_context, "session_id", None) or get_active_session_id()
        intent = self.record_intent(
            agent_name=agent_name,
            target_action=tool_name,
            rationale=f"Invoke tool '{tool_name}' with validated schema arguments.",
            inputs=args,
            expected_constraints={"requires_valid_schema": True},
            session_id=session_id,
        )
        _ACTIVE_TOOL_INTENT_ID.set(intent.intent_id)
        return None

    def adk_after_tool_callback(
        self,
        tool: Any,
        args: Dict[str, Any],
        tool_context: Any,
        tool_response: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """ADK `after_tool_callback` hook recording post-execution tool outcome."""
        tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
        intent_id = _ACTIVE_TOOL_INTENT_ID.get()
        _ACTIVE_TOOL_INTENT_ID.set(None)
        if not intent_id:
            for iid, rec in reversed(list(self._intents.items())):
                if rec.target_action in (tool_name, f"tool:{tool_name}"):
                    intent_id = iid
                    break
        if intent_id and intent_id not in self._outcomes:
            status = (
                str(tool_response.get("status") or tool_response.get("order_status") or "success")
                if isinstance(tool_response, dict)
                else "success"
            )
            satisfied = status.lower() not in ("error", "blocked_budget")
            self.record_outcome(
                intent_id,
                status=status,
                outputs=tool_response if isinstance(tool_response, dict) else {"result": tool_response},
                intent_satisfied=satisfied,
            )
        return None

    def get_paired_audit_trail(self) -> List[Dict[str, Any]]:
        """Return correlated pairs of (intent, outcome) for inspection and evaluation."""
        trail: List[Dict[str, Any]] = []
        for intent_id, intent in self._intents.items():
            outcome = self._outcomes.get(intent_id)
            intent_dict = asdict(intent)
            intent_dict.pop("started_perf_ns", None)
            trail.append(
                {
                    "intent_id": intent_id,
                    "intent": intent_dict,
                    "outcome": asdict(outcome) if outcome else None,
                }
            )
        return trail

    def clear(self) -> None:
        """Reset stored intents and outcomes."""
        self._intents.clear()
        self._outcomes.clear()
        _ACTIVE_TOOL_INTENT_ID.set(None)


_GLOBAL_AUDITOR = IntentOutcomeAuditor()


def get_global_auditor() -> IntentOutcomeAuditor:
    """Return the singleton `IntentOutcomeAuditor`."""
    return _GLOBAL_AUDITOR
