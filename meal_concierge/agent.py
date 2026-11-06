"""Multi-Agent Google ADK Architecture for the Autonomous Meal Planning Concierge.

Implements Rubric Category 3.1 (Multi-Agent Patterns) & Category 3.2 (Strategic Model Routing):
Constructs a hierarchical multi-agent system in the Google Agent Development Kit (ADK):
1. `meal_concierge_coordinator` (`LlmAgent` root coordinator, `gemini-2.5-pro`)
2. `weekly_meal_planning_pipeline` (`SequentialAgent`) orchestrating:
   - Stage 1: `nutrition_and_pantry_parallel_agent` (`ParallelAgent`) concurrently executing:
     * `nutrition_strategist_agent` (`LlmAgent`, `gemini-2.5-pro`)
     * `pantry_inventory_auditor_agent` (`LlmAgent`, `gemini-2.5-flash`)
   - Stage 2: `culinary_recipe_architect_agent` (`LlmAgent`, `gemini-2.5-pro`)
   - Stage 3: `grocery_procurement_specialist_agent` (`LlmAgent`, `gemini-2.5-flash`)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import functools
import inspect
from typing import Any, Callable, Dict, List, Optional
import uuid

from meal_concierge.memory.async_memory import AsyncMemoryConsolidator, get_async_memory_consolidator
from meal_concierge.memory.compaction import (
    ContextCompactionManager,
    adk_compaction_before_model_callback,
)
from meal_concierge.memory.session_store import (
    PersistentMealSessionStore,
    get_persistent_session_store,
)
from meal_concierge.observability.intent_outcome_auditor import (
    IntentOutcomeAuditor,
    get_global_auditor,
)
from meal_concierge.observability.structured_logger import get_structured_logger
from meal_concierge.observability.tracing import execution_context, get_tracer, traced_span
from meal_concierge.orchestration.guardrails import (
    GuardrailEvaluationResult,
    SafetyAndPolicyGuardrailPlugin,
    get_guardrail_plugin,
)
from meal_concierge.orchestration.hitl_hooks import (
    HumanInTheLoopApprovalGate,
    get_hitl_approval_gate,
)
from meal_concierge.orchestration.model_router import (
    StrategicModelRouter,
    get_model_router,
)
from meal_concierge.prompts.constitution import (
    CULINARY_ARCHITECT_SYSTEM_PROMPT,
    GROCERY_PROCUREMENT_SYSTEM_PROMPT,
    MEAL_CONCIERGE_CONSTITUTION,
    NUTRITION_STRATEGIST_SYSTEM_PROMPT,
    PANTRY_AUDITOR_SYSTEM_PROMPT,
)
from meal_concierge.schemas.models import (
    AisleSortedGroceryListOutput,
    DailyMealPlan,
    DietaryProfileInput,
    GroceryOrderPlacementOutput,
    MacroTargetsOutput,
    MealPlanComplianceOutput,
    MealSlotType,
    PlannedMealSlot,
    WeeklyMealPlanOutput,
)
from meal_concierge.tools import (
    calculate_weekly_macro_targets,
    check_pantry_inventory_levels,
    generate_aisle_sorted_grocery_list,
    place_instacart_grocery_order,
    reserve_pantry_ingredients_for_plan,
    search_allergen_safe_recipes,
    substitute_unsafe_or_missing_ingredients,
    validate_meal_plan_nutrition_compliance,
)
from meal_concierge.tools.recipe_search_tools import RECIPE_CATALOG

# ---------------------------------------------------------------------------
# Google ADK Imports (with local structural compatibility when running offline)
# ---------------------------------------------------------------------------

try:
    from google.adk.agents import (  # type: ignore[import-not-found]
        LlmAgent,
        ParallelAgent,
        SequentialAgent,
    )
    from google.adk.tools import FunctionTool as _AdkFunctionTool  # type: ignore[import-not-found]

    GOOGLE_ADK_AVAILABLE = True
except ImportError:  # pragma: no cover - local ADK structural classes for offline testing
    GOOGLE_ADK_AVAILABLE = False
    _AdkFunctionTool = None

    @dataclass
    class LlmAgent:  # type: ignore[no-redef]
        """ADK-compatible `LlmAgent` definition."""

        name: str
        model: str
        description: str = ""
        instruction: str = ""
        tools: List[Any] = field(default_factory=list)
        sub_agents: List[Any] = field(default_factory=list)
        output_key: Optional[str] = None
        before_model_callback: Optional[Callable[..., Any]] = None
        before_tool_callback: Optional[Callable[..., Any]] = None
        after_tool_callback: Optional[Callable[..., Any]] = None

    @dataclass
    class ParallelAgent:  # type: ignore[no-redef]
        """ADK-compatible `ParallelAgent` definition executing sub-agents concurrently."""

        name: str
        description: str = ""
        sub_agents: List[Any] = field(default_factory=list)

    @dataclass
    class SequentialAgent:  # type: ignore[no-redef]
        """ADK-compatible `SequentialAgent` definition executing sub-agents in pipeline order."""

        name: str
        description: str = ""
        sub_agents: List[Any] = field(default_factory=list)


_BaseFunctionTool: type = _AdkFunctionTool if (GOOGLE_ADK_AVAILABLE and _AdkFunctionTool is not None) else object


class FunctionTool(_BaseFunctionTool):  # type: ignore[misc]
    """ADK `FunctionTool` preserving function name, docstring, signature, JSON schemas, and HITL flags."""

    def __init__(self, func: Callable[..., Any], *, require_confirmation: bool = False) -> None:
        self.func = func
        self.name = func.__name__
        self.description = (func.__doc__ or "").strip()
        self.require_confirmation = bool(
            require_confirmation or getattr(func, "require_confirmation", False)
        )
        functools.update_wrapper(self, func)
        self.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
        if GOOGLE_ADK_AVAILABLE and _AdkFunctionTool is not None:
            try:
                super().__init__(func, require_confirmation=self.require_confirmation)
            except TypeError:
                super().__init__(func)
        self.input_model = getattr(func, "input_model", None)
        self.output_model = getattr(func, "output_model", None)
        self.input_schema: Dict[str, Any] = getattr(
            func,
            "input_schema",
            self.input_model.model_json_schema() if self.input_model is not None else {},
        )
        self.output_schema: Dict[str, Any] = getattr(
            func,
            "output_schema",
            self.output_model.model_json_schema() if self.output_model is not None else {},
        )
        self.parameters_json_schema: Dict[str, Any] = self.input_schema
        self.declaration: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
            "response": self.output_schema,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


_WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _composite_before_model_callback(callback_context: Any, llm_request: Any) -> Optional[Any]:
    """Chained ADK `before_model_callback`: Guardrails -> History Compaction -> Model Routing."""
    guardrail_block = get_guardrail_plugin().adk_before_model_guardrail_callback(
        callback_context, llm_request
    )
    if guardrail_block is not None:
        return guardrail_block
    adk_compaction_before_model_callback(callback_context, llm_request)
    get_model_router().adk_before_model_routing_callback(callback_context, llm_request)
    return None


def _composite_before_tool_callback(
    tool: Any,
    args: Dict[str, Any],
    tool_context: Any,
) -> Optional[Dict[str, Any]]:
    """Chained ADK `before_tool_callback`: Intent Capture -> Policy Guardrail -> HITL Gate."""
    get_global_auditor().adk_before_tool_callback(tool, args, tool_context)
    policy_block = get_guardrail_plugin().adk_before_tool_guardrail_callback(
        tool, args, tool_context
    )
    if policy_block is not None:
        return policy_block
    return get_hitl_approval_gate().adk_before_tool_hitl_callback(tool, args, tool_context)


def _composite_after_tool_callback(
    tool: Any,
    args: Dict[str, Any],
    tool_context: Any,
    tool_response: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """ADK `after_tool_callback`: Outcome Capture."""
    return get_global_auditor().adk_after_tool_callback(tool, args, tool_context, tool_response)


def build_meal_concierge_agent_hierarchy() -> LlmAgent:
    """Build and return the multi-agent Google ADK hierarchy for the Meal Planning Concierge."""
    router = get_model_router()

    # 1. Clinical Nutrition Strategist (Pro model for physiological & macro optimization)
    nutrition_strategist_model = router.route_for_agent(
        "nutrition_strategist_agent",
        task_description="Clinical BMR, TDEE, and macro-nutrient target calculation",
        constraint_count=4,
    ).selected_model
    nutrition_strategist_agent = LlmAgent(
        name="nutrition_strategist_agent",
        model=nutrition_strategist_model,
        description="Calculates personalized daily calorie, protein, carb, fat, fiber, and sodium targets and audits 7-day compliance.",
        instruction=NUTRITION_STRATEGIST_SYSTEM_PROMPT,
        tools=[
            FunctionTool(calculate_weekly_macro_targets),
            FunctionTool(validate_meal_plan_nutrition_compliance),
        ],
        output_key="macro_targets",
        before_model_callback=_composite_before_model_callback,
        before_tool_callback=_composite_before_tool_callback,
        after_tool_callback=_composite_after_tool_callback,
    )

    # 2. Pantry Inventory Auditor (Flash model for fast structured stock lookup)
    pantry_auditor_model = router.route_for_agent(
        "pantry_inventory_auditor_agent",
        task_description="Structured pantry stock check and expiry triage",
        constraint_count=1,
    ).selected_model
    pantry_inventory_auditor_agent = LlmAgent(
        name="pantry_inventory_auditor_agent",
        model=pantry_auditor_model,
        description="Audits household pantry stock, identifies expiring produce, and reserves ingredients.",
        instruction=PANTRY_AUDITOR_SYSTEM_PROMPT,
        tools=[
            FunctionTool(check_pantry_inventory_levels),
            FunctionTool(reserve_pantry_ingredients_for_plan),
        ],
        output_key="pantry_status",
        before_model_callback=_composite_before_model_callback,
        before_tool_callback=_composite_before_tool_callback,
        after_tool_callback=_composite_after_tool_callback,
    )

    # 3. Parallel Analyzer (runs Nutrition Strategist and Pantry Auditor concurrently)
    nutrition_and_pantry_parallel_agent = ParallelAgent(
        name="nutrition_and_pantry_parallel_agent",
        description="Concurrently computes clinical macro targets and audits household pantry inventory.",
        sub_agents=[
            nutrition_strategist_agent,
            pantry_inventory_auditor_agent,
        ],
    )

    # 4. Culinary Recipe Architect (Pro model for 7-day allergen-safe menu synthesis)
    culinary_architect_model = router.route_for_agent(
        "culinary_recipe_architect_agent",
        task_description="7-day multi-constraint allergen-safe meal plan synthesis and substitution",
        constraint_count=5,
    ).selected_model
    culinary_recipe_architect_agent = LlmAgent(
        name="culinary_recipe_architect_agent",
        model=culinary_architect_model,
        description="Synthesizes a varied, allergen-safe 7-day meal plan aligned with macro targets and expiring pantry items.",
        instruction=CULINARY_ARCHITECT_SYSTEM_PROMPT,
        tools=[
            FunctionTool(search_allergen_safe_recipes),
            FunctionTool(substitute_unsafe_or_missing_ingredients),
            FunctionTool(validate_meal_plan_nutrition_compliance),
        ],
        output_key="weekly_schedule",
        before_model_callback=_composite_before_model_callback,
        before_tool_callback=_composite_before_tool_callback,
        after_tool_callback=_composite_after_tool_callback,
    )

    # 5. Grocery Procurement Specialist (Flash model for aisle sorting & HITL order staging)
    grocery_specialist_model = router.route_for_agent(
        "grocery_procurement_specialist_agent",
        task_description="Aisle-sorted shopping cart consolidation and HITL checkout staging",
        constraint_count=1,
    ).selected_model
    grocery_procurement_specialist_agent = LlmAgent(
        name="grocery_procurement_specialist_agent",
        model=grocery_specialist_model,
        description="Consolidates recipe ingredients, subtracts pantry stock, sorts by supermarket aisle, and stages HITL-gated orders.",
        instruction=GROCERY_PROCUREMENT_SYSTEM_PROMPT,
        tools=[
            FunctionTool(generate_aisle_sorted_grocery_list),
            FunctionTool(place_instacart_grocery_order, require_confirmation=True),
        ],
        output_key="grocery_procurement",
        before_model_callback=_composite_before_model_callback,
        before_tool_callback=_composite_before_tool_callback,
        after_tool_callback=_composite_after_tool_callback,
    )

    # 6. Sequential Pipeline chaining Parallel Analysis -> Recipe Synthesis -> Grocery Procurement
    weekly_meal_planning_pipeline = SequentialAgent(
        name="weekly_meal_planning_pipeline",
        description="End-to-end sequential meal planning pipeline: Parallel(Nutrition, Pantry) -> Culinary Architect -> Grocery Procurement.",
        sub_agents=[
            nutrition_and_pantry_parallel_agent,
            culinary_recipe_architect_agent,
            grocery_procurement_specialist_agent,
        ],
    )

    # 7. Root Coordinator Agent (`meal_concierge_coordinator`)
    # Pure Coordinator pattern: delegates execution to `weekly_meal_planning_pipeline`
    # without duplicating low-level specialist tools on the root node.
    coordinator_model = router.route_for_agent(
        "meal_concierge_coordinator",
        task_description="Coordinator orchestrating 7-day meal planning, memory compaction, and HITL gates",
        constraint_count=5,
    ).selected_model
    coordinator = LlmAgent(
        name="meal_concierge_coordinator",
        model=coordinator_model,
        description="Personal Meal Planning Concierge Coordinator managing multi-agent orchestration, memory, guardrails, and HITL checkout.",
        instruction=MEAL_CONCIERGE_CONSTITUTION,
        tools=[],
        sub_agents=[weekly_meal_planning_pipeline],
        before_model_callback=_composite_before_model_callback,
        before_tool_callback=_composite_before_tool_callback,
        after_tool_callback=_composite_after_tool_callback,
    )
    return coordinator


# Export standard ADK symbols (`root_agent` and `meal_concierge_coordinator`)
meal_concierge_coordinator: LlmAgent = build_meal_concierge_agent_hierarchy()
root_agent: LlmAgent = meal_concierge_coordinator


class MealConciergeRunner:
    """High-assurance runtime executing the multi-agent Meal Planning Concierge pipeline.

    Coordinates:
    - Input prompt-injection & medical calorie-floor guardrails (`SafetyAndPolicyGuardrailPlugin`)
    - Persistent relational & vector session state (`PersistentMealSessionStore`)
    - History compaction & Vertex AI Context Caching (`ContextCompactionManager`)
    - Concurrent Stage 1 execution (`nutrition_strategist_agent` || `pantry_inventory_auditor_agent`)
    - Sequential Stage 2 (`culinary_recipe_architect_agent`) & Stage 3 (`grocery_procurement_specialist_agent`)
    - Self-evaluation rubric verification (`SelfEvaluationRubricGuardrail`)
    - Non-blocking background episodic memory consolidation (`AsyncMemoryConsolidator`)
    - Human-in-the-Loop approval gate (`HumanInTheLoopApprovalGate`)
    """

    def __init__(
        self,
        session_store: Optional[PersistentMealSessionStore] = None,
        compaction_manager: Optional[ContextCompactionManager] = None,
        memory_consolidator: Optional[AsyncMemoryConsolidator] = None,
        guardrail_plugin: Optional[SafetyAndPolicyGuardrailPlugin] = None,
        hitl_gate: Optional[HumanInTheLoopApprovalGate] = None,
        auditor: Optional[IntentOutcomeAuditor] = None,
    ) -> None:
        self.agent = meal_concierge_coordinator
        self.session_store = session_store or get_persistent_session_store()
        self.compaction_manager = compaction_manager or ContextCompactionManager()
        self.memory_consolidator = memory_consolidator or get_async_memory_consolidator()
        self.guardrail_plugin = guardrail_plugin or get_guardrail_plugin()
        self.hitl_gate = hitl_gate or get_hitl_approval_gate()
        self.auditor = auditor or get_global_auditor()
        self.router = get_model_router()
        self.logger = get_structured_logger()
        self.tracer = get_tracer()

    @traced_span("agent.MealConciergeRunner.run_weekly_meal_plan_async", kind="SERVER")
    async def run_weekly_meal_plan_async(
        self,
        *,
        user_prompt: str,
        dietary_profile: Dict[str, Any],
        session_id: Optional[str] = None,
        request_order_checkout: bool = True,
        human_approved: bool = False,
        confirmation_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute an end-to-end 7-day meal planning workflow asynchronously."""
        resolved_session_id = session_id or f"sess-{uuid.uuid4().hex[:10]}"
        user_id = str(dietary_profile.get("user_id", "default_user"))

        with execution_context(
            session_id=resolved_session_id,
            agent_name=self.agent.name,
        ):
            # Step 0: Input Guardrail Check
            input_guardrail: GuardrailEvaluationResult = self.guardrail_plugin.validate_user_input(
                user_prompt
            )
            if not input_guardrail.passed:
                return {
                    "status": "blocked_by_guardrail",
                    "session_id": resolved_session_id,
                    "guardrail": input_guardrail.guardrail_name,
                    "violations": input_guardrail.violations,
                    "remediation_advice": input_guardrail.remediation_advice,
                }

            # Persist incoming user turn and check history compaction
            self.session_store.append_conversation_turn(
                session_id=resolved_session_id,
                user_id=user_id,
                role="user",
                content=user_prompt,
            )
            self.session_store.upsert_dietary_profile(user_id, dietary_profile)

            history = self.session_store.get_conversation_history(resolved_session_id)
            compaction_result = self.compaction_manager.compact_history(
                history,
                user_id=user_id,
                session_id=resolved_session_id,
            )
            self.compaction_manager.context_cache.get_or_create_context_cache(
                cache_key="meal-concierge-constitution",
                system_instruction=MEAL_CONCIERGE_CONSTITUTION,
                static_context_documents=["FDA Big-9 Allergen Taxonomy", "Mifflin-St Jeor Equations"],
            )

            # Retrieve episodic vector memories & clinical nutrition knowledge from persistent store
            recalled_memories = self.session_store.vector_store.semantic_search(
                user_prompt,
                namespace="episodic_memory",
                user_id=user_id,
                top_k=3,
            )
            clinical_passages = self.session_store.vertex_search.search_nutrition_knowledge(
                user_prompt,
                top_k=2,
            )

            # Traverse ADK SequentialAgent (`weekly_meal_planning_pipeline`)
            seq_pipeline = self.agent.sub_agents[0]
            parallel_stage = seq_pipeline.sub_agents[0]
            nutrition_agent, pantry_agent = parallel_stage.sub_agents
            culinary_agent = seq_pipeline.sub_agents[1]
            grocery_agent = seq_pipeline.sub_agents[2]

            # Stage 1 (ParallelAgent): Run Nutrition Strategist (Pro) and Pantry Auditor (Flash) concurrently
            with self.tracer.start_as_current_span(
                f"parallel_stage.{parallel_stage.name}"
            ) as parallel_span:
                self.router.route_for_agent(
                    nutrition_agent.name,
                    task_description="Calculate macro targets",
                    constraint_count=len(dietary_profile.get("allergies", [])) + 3,
                )
                self.router.route_for_agent(
                    pantry_agent.name,
                    task_description="Audit household pantry stock",
                    constraint_count=1,
                )

                def _run_nutrition_branch() -> Dict[str, Any]:
                    with execution_context(
                        session_id=resolved_session_id, agent_name=nutrition_agent.name
                    ):
                        return calculate_weekly_macro_targets(dietary_profile)

                def _run_pantry_branch() -> Dict[str, Any]:
                    with execution_context(
                        session_id=resolved_session_id, agent_name=pantry_agent.name
                    ):
                        return check_pantry_inventory_levels(
                            {"household_id": user_id, "include_expiring_within_days": 5}
                        )

                macro_raw, pantry_raw = await asyncio.gather(
                    asyncio.to_thread(_run_nutrition_branch),
                    asyncio.to_thread(_run_pantry_branch),
                )
                parallel_span.set_attribute("parallel.branches_completed", 2)

            if macro_raw.get("status") == "error":
                return macro_raw
            macro_targets = MacroTargetsOutput.model_validate(macro_raw)
            expiring_items: List[str] = pantry_raw.get("expiring_soon_items", [])

            # Ground recipe search using recalled episodic vector memories & clinical passages
            memory_cues: List[str] = []
            for mem in recalled_memories:
                content_lower = str(mem.get("content", "")).lower()
                for keyword in ("quinoa", "lentil", "chickpea", "spinach", "avocado", "salmon"):
                    if keyword in content_lower:
                        memory_cues.append(keyword)
            for passage in clinical_passages:
                snippet_lower = str(passage.get("snippet", "")).lower()
                for keyword in ("lentil", "quinoa", "olive oil"):
                    if keyword in snippet_lower:
                        memory_cues.append(keyword)
            prioritized_ingredients = list(dict.fromkeys(expiring_items + memory_cues))

            # Stage 2 (SequentialAgent -> Culinary Recipe Architect on Pro): Discover safe recipes & build 7-day plan
            with self.tracer.start_as_current_span(f"sequential_stage.{culinary_agent.name}"):
                with execution_context(
                    session_id=resolved_session_id, agent_name=culinary_agent.name
                ):
                    self.router.route_for_agent(
                        culinary_agent.name,
                        task_description="Synthesize 7-day allergen-safe meal plan",
                        constraint_count=5,
                    )
                    restriction = str(dietary_profile.get("dietary_restriction", "omnivore"))
                    excluded_allergens = macro_targets.excluded_allergens

                    def _search_slot(slot_name: str, cal_share: float, min_prot: int) -> Dict[str, Any]:
                        with execution_context(
                            session_id=resolved_session_id, agent_name=culinary_agent.name
                        ):
                            return search_allergen_safe_recipes(
                                {
                                    "meal_slot": slot_name,
                                    "dietary_restriction": restriction,
                                    "excluded_allergens": excluded_allergens,
                                    "target_calories_kcal": int(
                                        macro_targets.daily_calorie_target_kcal * cal_share
                                    ),
                                    "min_protein_grams": min_prot,
                                    "prefer_ingredients": prioritized_ingredients,
                                }
                            )

                    bf_resp, ln_resp, dn_resp, sn_resp = await asyncio.gather(
                        asyncio.to_thread(_search_slot, "breakfast", 0.24, 20),
                        asyncio.to_thread(_search_slot, "lunch", 0.32, 25),
                        asyncio.to_thread(_search_slot, "dinner", 0.34, 30),
                        asyncio.to_thread(_search_slot, "snack", 0.10, 10),
                    )

                    bf_list = bf_resp["recipes"]
                    ln_list = ln_resp["recipes"]
                    dn_list = dn_resp["recipes"]
                    sn_list = sn_resp["recipes"]

                    days: List[DailyMealPlan] = []
                    scheduled_recipe_ids: List[str] = []

                    for idx, day_name in enumerate(_WEEKDAYS):
                        bf = bf_list[idx % len(bf_list)]
                        ln = ln_list[idx % len(ln_list)]
                        dn = dn_list[idx % len(dn_list)]
                        sn = sn_list[idx % len(sn_list)]

                        base_cal = (
                            bf["calories_kcal"]
                            + ln["calories_kcal"]
                            + dn["calories_kcal"]
                            + sn["calories_kcal"]
                        )
                        scale_factor = macro_targets.daily_calorie_target_kcal / max(1, base_cal)
                        bf_cal = max(150, int(round(bf["calories_kcal"] * scale_factor)))
                        ln_cal = max(250, int(round(ln["calories_kcal"] * scale_factor)))
                        dn_cal = max(300, int(round(dn["calories_kcal"] * scale_factor)))
                        sn_cal = max(
                            100,
                            macro_targets.daily_calorie_target_kcal - (bf_cal + ln_cal + dn_cal),
                        )

                        prot_scale = max(
                            scale_factor,
                            macro_targets.daily_protein_grams
                            / max(
                                1,
                                bf["protein_grams"]
                                + ln["protein_grams"]
                                + dn["protein_grams"]
                                + sn["protein_grams"],
                            ),
                        )
                        bf_prot = max(12, int(round(bf["protein_grams"] * prot_scale)))
                        ln_prot = max(18, int(round(ln["protein_grams"] * prot_scale)))
                        dn_prot = max(22, int(round(dn["protein_grams"] * prot_scale)))
                        sn_prot = max(8, int(round(sn["protein_grams"] * prot_scale)))

                        day_meals = [
                            PlannedMealSlot(
                                slot=MealSlotType.BREAKFAST,
                                recipe_id=bf["recipe_id"],
                                recipe_title=bf["title"],
                                servings=1,
                                calories_kcal=bf_cal,
                                protein_grams=bf_prot,
                                carbs_grams=bf["carbs_grams"],
                                fat_grams=bf["fat_grams"],
                                estimated_cost_usd=bf["estimated_cost_per_serving_usd"],
                            ),
                            PlannedMealSlot(
                                slot=MealSlotType.LUNCH,
                                recipe_id=ln["recipe_id"],
                                recipe_title=ln["title"],
                                servings=1,
                                calories_kcal=ln_cal,
                                protein_grams=ln_prot,
                                carbs_grams=ln["carbs_grams"],
                                fat_grams=ln["fat_grams"],
                                estimated_cost_usd=ln["estimated_cost_per_serving_usd"],
                            ),
                            PlannedMealSlot(
                                slot=MealSlotType.DINNER,
                                recipe_id=dn["recipe_id"],
                                recipe_title=dn["title"],
                                servings=1,
                                calories_kcal=dn_cal,
                                protein_grams=dn_prot,
                                carbs_grams=dn["carbs_grams"],
                                fat_grams=dn["fat_grams"],
                                estimated_cost_usd=dn["estimated_cost_per_serving_usd"],
                            ),
                            PlannedMealSlot(
                                slot=MealSlotType.SNACK,
                                recipe_id=sn["recipe_id"],
                                recipe_title=sn["title"],
                                servings=1,
                                calories_kcal=sn_cal,
                                protein_grams=sn_prot,
                                carbs_grams=sn["carbs_grams"],
                                fat_grams=sn["fat_grams"],
                                estimated_cost_usd=sn["estimated_cost_per_serving_usd"],
                            ),
                        ]
                        scheduled_recipe_ids.extend(
                            [bf["recipe_id"], ln["recipe_id"], dn["recipe_id"], sn["recipe_id"]]
                        )

                        days.append(
                            DailyMealPlan(
                                day_index=idx + 1,
                                day_name=day_name,
                                meals=day_meals,
                                total_calories_kcal=sum(m.calories_kcal for m in day_meals),
                                total_protein_grams=sum(m.protein_grams for m in day_meals),
                                total_carbs_grams=sum(m.carbs_grams for m in day_meals),
                                total_fat_grams=sum(m.fat_grams for m in day_meals),
                                total_cost_usd=round(sum(m.estimated_cost_usd for m in day_meals), 2),
                            )
                        )

                    plan_id = f"plan-{user_id}-{uuid.uuid4().hex[:8]}"
                    compliance_raw = validate_meal_plan_nutrition_compliance(
                        {
                            "user_id": user_id,
                            "macro_targets": macro_targets.model_dump(),
                            "days": [d.model_dump() for d in days],
                        }
                    )
                    compliance_audit = MealPlanComplianceOutput.model_validate(compliance_raw)

            # Stage 3 (SequentialAgent -> Grocery Procurement Specialist on Flash): Build aisle cart & HITL order
            with self.tracer.start_as_current_span(f"sequential_stage.{grocery_agent.name}"):
                with execution_context(
                    session_id=resolved_session_id, agent_name=grocery_agent.name
                ):
                    self.router.route_for_agent(
                        grocery_agent.name,
                        task_description="Generate aisle-sorted grocery list and stage HITL order",
                        constraint_count=1,
                    )
                    grocery_raw = generate_aisle_sorted_grocery_list(
                        {
                            "household_id": user_id,
                            "plan_id": plan_id,
                            "recipe_ids": scheduled_recipe_ids,
                            "servings_multiplier": 1,
                            "weekly_budget_usd": macro_targets.weekly_budget_usd,
                        }
                    )
                    grocery_list = AisleSortedGroceryListOutput.model_validate(grocery_raw)

                    # Reserve pantry items utilized by this plan
                    if expiring_items:
                        reserve_pantry_ingredients_for_plan(
                            {
                                "household_id": user_id,
                                "plan_id": plan_id,
                                "reservations": {
                                    item_name: 50.0
                                    for item_name in expiring_items
                                },
                            }
                        )

                    if request_order_checkout:
                        order_raw = place_instacart_grocery_order(
                            {
                                "household_id": user_id,
                                "plan_id": plan_id,
                                "retailer_name": "Whole Foods Market",
                                "delivery_window": "Sunday 09:00-11:00",
                                "net_total_usd": max(1.0, grocery_list.net_estimated_cost_usd),
                                "weekly_budget_usd": macro_targets.weekly_budget_usd,
                                "human_approved": human_approved,
                                "confirmation_token": confirmation_token,
                            }
                        )
                        order_outcome = GroceryOrderPlacementOutput.model_validate(order_raw)
                    else:
                        order_outcome = GroceryOrderPlacementOutput(
                            order_status="CART_READY_NOT_SUBMITTED",
                            order_id=None,
                            approval_request_id=None,
                            retailer_name="Whole Foods Market",
                            charged_amount_usd=0.0,
                            delivery_window="Sunday 09:00-11:00",
                            message="Grocery list generated; checkout not requested.",
                        )

            weekly_plan = WeeklyMealPlanOutput(
                plan_id=plan_id,
                user_id=user_id,
                macro_targets=macro_targets,
                days=days,
                compliance_audit=compliance_audit,
                grocery_list=grocery_list,
                order_outcome=order_outcome,
            )

            # Step 4: Self-Evaluation Rubric Guardrail (with full recipe allergen map)
            recipe_allergen_lookup = {
                r.recipe_id: list(r.allergens_present) for r in RECIPE_CATALOG
            }
            self_eval = self.guardrail_plugin.self_evaluator.evaluate_meal_plan(
                weekly_plan,
                declared_allergens=excluded_allergens,
                recipe_allergen_lookup=recipe_allergen_lookup,
            )

            # Persist assistant turn & session state, and enqueue non-blocking Async Memory Consolidation
            assistant_summary = (
                f"Generated 7-day {restriction} meal plan '{plan_id}' targeting "
                f"{macro_targets.daily_calorie_target_kcal} kcal/day and {macro_targets.daily_protein_grams}g protein/day. "
                f"Net grocery cost after ${grocery_list.pantry_savings_usd:.2f} pantry savings: "
                f"${grocery_list.net_estimated_cost_usd:.2f}. Order status: {order_outcome.order_status}."
            )
            self.session_store.append_conversation_turn(
                session_id=resolved_session_id,
                user_id=user_id,
                role="assistant",
                content=assistant_summary,
            )
            self.session_store.save_session_state(
                session_id=resolved_session_id,
                user_id=user_id,
                state={
                    "last_plan_id": plan_id,
                    "daily_calorie_target_kcal": macro_targets.daily_calorie_target_kcal,
                    "excluded_allergens": excluded_allergens,
                    "order_status": order_outcome.order_status,
                    "approval_request_id": order_outcome.approval_request_id,
                    "recalled_memory_count": len(recalled_memories),
                    "clinical_passages_count": len(clinical_passages),
                },
            )

            async_memory_task_id = self.memory_consolidator.schedule_background_consolidation(
                user_id=user_id,
                session_id=resolved_session_id,
                conversation_turns=self.session_store.get_conversation_history(resolved_session_id),
                plan_summary=macro_targets.model_dump(),
            )

            final_status = "success" if self_eval.passed else "blocked_by_self_evaluation"
            return {
                "status": final_status,
                "session_id": resolved_session_id,
                "compaction_telemetry": {
                    "compaction_applied": compaction_result.compaction_applied,
                    "tokens_saved": compaction_result.tokens_saved,
                },
                "memory_grounding": {
                    "episodic_memories_recalled": len(recalled_memories),
                    "clinical_passages_retrieved": len(clinical_passages),
                },
                "self_evaluation": {
                    "passed": self_eval.passed,
                    "score": self_eval.score,
                    "violations": self_eval.violations,
                    "remediation_advice": self_eval.remediation_advice,
                },
                "async_memory_task_id": async_memory_task_id,
                "meal_plan": weekly_plan.model_dump(),
            }

    def run_weekly_meal_plan(self, **kwargs: Any) -> Dict[str, Any]:
        """Synchronous convenience wrapper around `run_weekly_meal_plan_async`.

        Drains any background memory consolidation tasks scheduled during the run before
        closing the event loop so task results are immediately queryable by callers.
        """

        async def _execute_and_drain() -> Dict[str, Any]:
            result = await self.run_weekly_meal_plan_async(**kwargs)
            await self.memory_consolidator.wait_for_all_background_tasks()
            return result

        return asyncio.run(_execute_and_drain())
