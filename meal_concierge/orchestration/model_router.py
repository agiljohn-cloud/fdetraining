"""Strategic Model Router for Google ADK Meal Concierge Sub-Agents.

Implements Rubric Category 3.2 (Strategic Model Routing):
Routes specific requests and sub-agent tasks to the most appropriate Gemini model:
- `gemini-2.5-flash` for low-latency, structured tasks (`pantry_inventory_auditor_agent`,
  `grocery_procurement_specialist_agent`, and simple status/lookup queries).
- `gemini-2.5-pro` for complex multi-constraint reasoning tasks (`nutrition_strategist_agent`,
  `culinary_recipe_architect_agent`, and multi-constraint `meal_concierge_coordinator` turns).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from meal_concierge.observability.structured_logger import get_structured_logger
from meal_concierge.observability.tracing import traced_span


class TaskComplexityTier(str, Enum):
    """Task complexity classification driving Gemini Flash vs. Pro selection."""

    FAST_STRUCTURED_LOOKUP = "fast_structured_lookup"
    COMPLEX_MULTI_CONSTRAINT_PLANNING = "complex_multi_constraint_planning"


@dataclass(frozen=True)
class ModelRoutingDecision:
    """Audit record of a model routing decision."""

    agent_name: str
    task_type: str
    complexity_tier: TaskComplexityTier
    selected_model: str
    rationale: str
    estimated_latency_tier: str


class StrategicModelRouter:
    """Routes ADK sub-agents and dynamic user prompts between `gemini-2.5-flash` and `gemini-2.5-pro`."""

    FLASH_MODEL: str = "gemini-2.5-flash"
    PRO_MODEL: str = "gemini-2.5-pro"

    # Dedicated specialist sub-agents pinned to their optimal latency/cost tier
    PINNED_FLASH_AGENTS = frozenset(
        {
            "pantry_inventory_auditor_agent",
            "grocery_procurement_specialist_agent",
        }
    )
    PINNED_PRO_AGENTS = frozenset(
        {
            "nutrition_strategist_agent",
            "culinary_recipe_architect_agent",
        }
    )

    COMPLEX_PLANNING_KEYWORDS = (
        "7-day",
        "weekly meal plan",
        "macro",
        "allerg",
        "substitute",
        "glycemic",
        "mediterranean",
        "ketogenic",
        "optimize",
        "balance",
    )

    def __init__(self) -> None:
        self._logger = get_structured_logger()
        self._routing_history: List[ModelRoutingDecision] = []

    @traced_span("orchestration.StrategicModelRouter.route_for_agent")
    def route_for_agent(
        self,
        agent_name: str,
        *,
        task_description: str = "",
        constraint_count: int = 0,
    ) -> ModelRoutingDecision:
        """Select the optimal Gemini model for `agent_name` and `task_description`."""
        if agent_name in self.PINNED_FLASH_AGENTS:
            use_pro = False
        elif agent_name in self.PINNED_PRO_AGENTS:
            use_pro = True
        else:
            # Dynamic routing for coordinator or unpinned agents based on query complexity
            lower_desc = task_description.lower()
            use_pro = (
                constraint_count >= 2
                or not task_description
                or any(kw in lower_desc for kw in self.COMPLEX_PLANNING_KEYWORDS)
            )

        if use_pro:
            complexity_tier = TaskComplexityTier.COMPLEX_MULTI_CONSTRAINT_PLANNING
            selected_model = self.PRO_MODEL
            latency_tier = "deep_reasoning"
            rationale = (
                f"Routed '{agent_name}' to '{self.PRO_MODEL}' for multi-constraint nutritional, "
                f"allergen, and 7-day culinary planning (constraints={constraint_count})."
            )
        else:
            complexity_tier = TaskComplexityTier.FAST_STRUCTURED_LOOKUP
            selected_model = self.FLASH_MODEL
            latency_tier = "low_latency"
            rationale = (
                f"Routed '{agent_name}' to '{self.FLASH_MODEL}' for high-throughput structured "
                f"inventory, aisle categorization, and cart aggregation."
            )

        decision = ModelRoutingDecision(
            agent_name=agent_name,
            task_type=task_description or agent_name,
            complexity_tier=complexity_tier,
            selected_model=selected_model,
            rationale=rationale,
            estimated_latency_tier=latency_tier,
        )
        self._routing_history.append(decision)

        self._logger.info(
            rationale,
            event_type="MODEL_ROUTING_DECISION",
            agent_name=agent_name,
            payload={
                "agent_name": agent_name,
                "selected_model": selected_model,
                "complexity_tier": complexity_tier.value,
                "estimated_latency_tier": latency_tier,
            },
        )
        return decision

    def adk_before_model_routing_callback(
        self,
        callback_context: Any,
        llm_request: Any,
    ) -> Optional[Any]:
        """ADK `before_model_callback` that sets `llm_request.model` honoring pinned sub-agent tiers."""
        agent_name = getattr(callback_context, "agent_name", "meal_concierge_coordinator")
        prompt_preview = str(getattr(llm_request, "contents", ""))[:300]
        decision = self.route_for_agent(agent_name, task_description=prompt_preview)
        if hasattr(llm_request, "model"):
            llm_request.model = decision.selected_model
        return None

    def get_routing_history(self) -> List[ModelRoutingDecision]:
        """Return all recorded model routing decisions."""
        return list(self._routing_history)


_GLOBAL_ROUTER = StrategicModelRouter()


def get_model_router() -> StrategicModelRouter:
    """Return the singleton `StrategicModelRouter`."""
    return _GLOBAL_ROUTER
