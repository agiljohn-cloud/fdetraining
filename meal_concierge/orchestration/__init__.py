"""Orchestration: Strategic Model Routing, Safety/Self-Eval Guardrail Plugins, and HITL Hooks."""

from meal_concierge.orchestration.guardrails import (
    GuardrailEvaluationResult,
    SafetyAndPolicyGuardrailPlugin,
    SelfEvaluationRubricGuardrail,
    get_guardrail_plugin,
)
from meal_concierge.orchestration.hitl_hooks import (
    ApprovalRequestTicket,
    HumanApprovalRequiredError,
    HumanInTheLoopApprovalGate,
    get_hitl_approval_gate,
)
from meal_concierge.orchestration.model_router import (
    ModelRoutingDecision,
    StrategicModelRouter,
    TaskComplexityTier,
    get_model_router,
)

__all__ = [
    "ApprovalRequestTicket",
    "GuardrailEvaluationResult",
    "HumanApprovalRequiredError",
    "HumanInTheLoopApprovalGate",
    "ModelRoutingDecision",
    "SafetyAndPolicyGuardrailPlugin",
    "SelfEvaluationRubricGuardrail",
    "StrategicModelRouter",
    "TaskComplexityTier",
    "get_guardrail_plugin",
    "get_hitl_approval_gate",
    "get_model_router",
]
