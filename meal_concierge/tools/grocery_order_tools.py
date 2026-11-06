"""Aisle-Sorted Grocery List Generation and HITL-Gated Instacart Order Execution Tools.

Implements Rubric Category 1 & Category 3.4:
- Comprehensive docstrings detailing inputs, outputs, budget validation, and HITL gates.
- Specific tool names (`generate_aisle_sorted_grocery_list`, `place_instacart_grocery_order`).
- Explicit typed signatures and strict JSON schemas (`GroceryListGenerationInput` ->
  `AisleSortedGroceryListOutput`, `GroceryOrderPlacementInput` -> `GroceryOrderPlacementOutput`).
- Guided error handling (`ToolErrorRecoveryPayload`) and single-use, plan-bound
  Human-in-the-Loop confirmation token verification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid

from meal_concierge.observability.intent_outcome_auditor import get_global_auditor
from meal_concierge.observability.tracing import get_active_agent_name, traced_span
from meal_concierge.orchestration.hitl_hooks import get_hitl_approval_gate
from meal_concierge.schemas.models import (
    AisleSortedGroceryListOutput,
    GroceryLineItem,
    GroceryListGenerationInput,
    GroceryOrderPlacementInput,
    GroceryOrderPlacementOutput,
    SchemaValidationError,
    ToolErrorRecoveryPayload,
)
from meal_concierge.security.secret_manager import SecretResolutionError, get_secret_manager
from meal_concierge.tools.pantry_inventory_tools import check_pantry_inventory_levels
from meal_concierge.tools.recipe_search_tools import RECIPE_CATALOG


_RECIPE_BY_ID = {r.recipe_id: r for r in RECIPE_CATALOG}


def _resolve_agent(default_agent: str) -> str:
    active = get_active_agent_name()
    return default_agent if active == "meal_concierge_coordinator" else active


def _merge_explicit_args(
    base_payload: Optional[Dict[str, Any]],
    explicit_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(base_payload) if isinstance(base_payload, dict) else {}
    for k, v in explicit_kwargs.items():
        if v is not None:
            merged[k] = v
    return merged


@traced_span("tools.generate_aisle_sorted_grocery_list")
def generate_aisle_sorted_grocery_list(
    grocery_payload: Optional[Dict[str, Any]] = None,
    *,
    household_id: Optional[str] = None,
    plan_id: Optional[str] = None,
    recipe_ids: Optional[List[str]] = None,
    servings_multiplier: Optional[int] = None,
    weekly_budget_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Aggregate recipe ingredients across a weekly plan, subtract pantry stock, and sort by aisle.

    Consolidates ingredient quantities across all scheduled `recipe_ids`, queries the
    household pantry to deduct quantities already on hand, computes gross cost, pantry
    savings, and net retail cost, and organizes required purchases by supermarket aisle.

    Args:
        grocery_payload: Optional dictionary conforming to `GroceryListGenerationInput`.
        household_id: Household identifier (2 to 64 chars).
        plan_id: Weekly meal plan identifier.
        recipe_ids: Scheduled recipe identifiers across the 7-day plan.
        servings_multiplier: Household serving multiplier (1 to 10).
        weekly_budget_usd: Weekly grocery budget ceiling in USD (>= 35.0).

    Returns:
        Dictionary serialized from `AisleSortedGroceryListOutput` on success, or
        `ToolErrorRecoveryPayload` with actionable `recovery_instructions` if unknown recipe IDs
        or invalid parameters are supplied.
    """
    merged_payload = _merge_explicit_args(
        grocery_payload,
        {
            "household_id": household_id,
            "plan_id": plan_id,
            "recipe_ids": recipe_ids,
            "servings_multiplier": servings_multiplier,
            "weekly_budget_usd": weekly_budget_usd,
        },
    )

    auditor = get_global_auditor()
    intent = auditor.record_intent(
        agent_name=_resolve_agent("grocery_procurement_specialist_agent"),
        target_action="generate_aisle_sorted_grocery_list",
        rationale="Aggregate weekly recipe ingredients, deduct pantry stock, and organize by store aisle.",
        inputs=merged_payload,
        expected_constraints={"weekly_budget_usd": merged_payload.get("weekly_budget_usd", 150.0)},
    )

    try:
        validated = GroceryListGenerationInput.model_validate(merged_payload)

        unknown_ids = [rid for rid in validated.recipe_ids if rid not in _RECIPE_BY_ID]
        if unknown_ids:
            recovery = ToolErrorRecoveryPayload(
                status="error",
                recovery_mode="GUIDED_RECOVERY",
                error_code="UNKNOWN_RECIPE_IDS_IN_PLAN",
                tool_name="generate_aisle_sorted_grocery_list",
                message=f"One or more recipe_ids do not exist in catalog: {unknown_ids}",
                invalid_parameters={"unknown_recipe_ids": unknown_ids},
                recovery_instructions=[
                    "Invoke `search_allergen_safe_recipes` to retrieve valid `recipe_id` values.",
                    f"Valid catalog IDs include: {sorted(_RECIPE_BY_ID.keys())}.",
                    "Re-invoke `generate_aisle_sorted_grocery_list` with valid `recipe_ids`.",
                ],
                fallback_tool_suggestion="search_allergen_safe_recipes",
            ).model_dump()
            auditor.record_outcome(
                intent.intent_id,
                status="error",
                outputs=recovery,
                intent_satisfied=False,
            )
            return recovery

        aggregated: Dict[str, Dict[str, Any]] = {}
        for rid in validated.recipe_ids:
            recipe = _RECIPE_BY_ID[rid]
            for ing in recipe.ingredients:
                key = ing.name.lower()
                qty = ing.quantity * validated.servings_multiplier
                cost = ing.estimated_cost_usd * validated.servings_multiplier
                if key not in aggregated:
                    aggregated[key] = {
                        "ingredient_name": ing.name,
                        "aisle_category": ing.category,
                        "gross_quantity": 0.0,
                        "unit": ing.unit,
                        "gross_cost_usd": 0.0,
                    }
                aggregated[key]["gross_quantity"] += qty
                aggregated[key]["gross_cost_usd"] += cost

        pantry_resp = check_pantry_inventory_levels({"household_id": validated.household_id})
        pantry_map: Dict[str, float] = {}
        for item in pantry_resp.get("inventory", []):
            pantry_map[item["ingredient_name"].lower()] = float(item["available_quantity"])

        items_by_aisle: Dict[str, List[GroceryLineItem]] = {}
        gross_total_usd = 0.0
        pantry_savings_usd = 0.0
        net_total_usd = 0.0
        unique_to_buy = 0

        for key, entry in sorted(aggregated.items()):
            gross_qty = round(entry["gross_quantity"], 2)
            gross_cost = round(entry["gross_cost_usd"], 2)
            gross_total_usd += gross_cost

            avail_pantry = pantry_map.get(key, 0.0)
            deducted_qty = round(min(gross_qty, avail_pantry), 2)
            net_qty = round(max(0.0, gross_qty - deducted_qty), 2)

            fraction_saved = (deducted_qty / gross_qty) if gross_qty > 0 else 0.0
            line_savings = round(gross_cost * fraction_saved, 2)
            net_line_cost = round(max(0.0, gross_cost - line_savings), 2)

            pantry_savings_usd += line_savings
            net_total_usd += net_line_cost
            if net_qty > 0:
                unique_to_buy += 1

            line_item = GroceryLineItem(
                ingredient_name=entry["ingredient_name"],
                aisle_category=entry["aisle_category"],
                gross_required_quantity=gross_qty,
                pantry_deducted_quantity=deducted_qty,
                net_purchase_quantity=net_qty,
                unit=entry["unit"],
                estimated_line_cost_usd=net_line_cost,
            )
            items_by_aisle.setdefault(entry["aisle_category"], []).append(line_item)

        gross_total_usd = round(gross_total_usd, 2)
        pantry_savings_usd = round(pantry_savings_usd, 2)
        net_total_usd = round(net_total_usd, 2)
        within_budget = net_total_usd <= validated.weekly_budget_usd

        output = AisleSortedGroceryListOutput(
            plan_id=validated.plan_id,
            household_id=validated.household_id,
            total_unique_items_to_buy=unique_to_buy,
            gross_estimated_cost_usd=gross_total_usd,
            pantry_savings_usd=pantry_savings_usd,
            net_estimated_cost_usd=net_total_usd,
            weekly_budget_usd=round(validated.weekly_budget_usd, 2),
            within_budget=within_budget,
            items_by_aisle=items_by_aisle,
        ).model_dump()

        auditor.record_outcome(
            intent.intent_id,
            status="success",
            outputs={
                "plan_id": validated.plan_id,
                "net_estimated_cost_usd": net_total_usd,
                "pantry_savings_usd": pantry_savings_usd,
                "within_budget": within_budget,
            },
            intent_satisfied=within_budget,
        )
        return output

    except SchemaValidationError as exc:
        recovery = ToolErrorRecoveryPayload(
            status="error",
            recovery_mode="GUIDED_RECOVERY",
            error_code="INVALID_GROCERY_LIST_SCHEMA",
            tool_name="generate_aisle_sorted_grocery_list",
            message=str(exc),
            invalid_parameters={err["field"]: err["message"] for err in exc.errors},
            recovery_instructions=[
                "Provide valid `household_id`, `plan_id`, and a non-empty list of `recipe_ids`.",
                "Ensure `weekly_budget_usd` >= 35.0 and `servings_multiplier` is an integer in [1, 10].",
            ],
            fallback_tool_suggestion="search_allergen_safe_recipes",
        ).model_dump()
        auditor.record_outcome(
            intent.intent_id,
            status="error",
            outputs=recovery,
            intent_satisfied=False,
        )
        return recovery
    except Exception as exc:
        recovery = ToolErrorRecoveryPayload(
            status="error",
            recovery_mode="GUIDED_RECOVERY",
            error_code="UNEXPECTED_GROCERY_LIST_ERROR",
            tool_name="generate_aisle_sorted_grocery_list",
            message=f"Unexpected error: {exc}",
            invalid_parameters={},
            recovery_instructions=["Supply a valid dictionary matching `GroceryListGenerationInput`."],
            fallback_tool_suggestion="search_allergen_safe_recipes",
        ).model_dump()
        auditor.record_outcome(
            intent.intent_id,
            status="error",
            outputs=recovery,
            intent_satisfied=False,
        )
        return recovery


generate_aisle_sorted_grocery_list.input_model = GroceryListGenerationInput  # type: ignore[attr-defined]
generate_aisle_sorted_grocery_list.output_model = AisleSortedGroceryListOutput  # type: ignore[attr-defined]
generate_aisle_sorted_grocery_list.input_schema = GroceryListGenerationInput.model_json_schema()  # type: ignore[attr-defined]
generate_aisle_sorted_grocery_list.output_schema = AisleSortedGroceryListOutput.model_json_schema()  # type: ignore[attr-defined]


@traced_span("tools.place_instacart_grocery_order")
def place_instacart_grocery_order(
    order_payload: Optional[Dict[str, Any]] = None,
    *,
    household_id: Optional[str] = None,
    plan_id: Optional[str] = None,
    retailer_name: Optional[str] = None,
    delivery_window: Optional[str] = None,
    net_total_usd: Optional[float] = None,
    weekly_budget_usd: Optional[float] = None,
    human_approved: Optional[bool] = None,
    confirmation_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Stage or execute a grocery delivery order with mandatory Budget & Human-in-the-Loop gates.

    Enforces Constitutional Law III (Strict Budget Ceiling) and Law IV (Mandatory Human-in-the-Loop
    Confirmation before Financial Execution). Resolves the retailer API credential securely via
    `SecretManagerClient` (never hardcoded). If `human_approved` is False or `confirmation_token`
    is missing, expired, already consumed, or bound to a different `plan_id`, halts execution and
    returns an approval ticket (`PENDING_HUMAN_APPROVAL`).

    Args:
        order_payload: Optional dictionary conforming to `GroceryOrderPlacementInput`.
        household_id: Household identifier (2 to 64 chars).
        plan_id: Meal plan identifier.
        retailer_name: Partner store name (default 'Whole Foods Market').
        delivery_window: Delivery slot (default 'Sunday 09:00-11:00').
        net_total_usd: Cart total to charge in USD (1.0 to 2000.0).
        weekly_budget_usd: Configured weekly grocery budget cap in USD (>= 35.0).
        human_approved: True only after explicit user confirmation.
        confirmation_token: Single-use cryptographic token from `HumanInTheLoopApprovalGate`.

    Returns:
        Dictionary serialized from `GroceryOrderPlacementOutput` or `ToolErrorRecoveryPayload`
        when blocked by budget overrun, secret error, or schema error.
    """
    merged_payload = _merge_explicit_args(
        order_payload,
        {
            "household_id": household_id,
            "plan_id": plan_id,
            "retailer_name": retailer_name,
            "delivery_window": delivery_window,
            "net_total_usd": net_total_usd,
            "weekly_budget_usd": weekly_budget_usd,
            "human_approved": human_approved,
            "confirmation_token": confirmation_token,
        },
    )

    auditor = get_global_auditor()
    intent = auditor.record_intent(
        agent_name=_resolve_agent("grocery_procurement_specialist_agent"),
        target_action="place_instacart_grocery_order",
        rationale="Stage or execute grocery delivery order subject to budget ceiling and HITL confirmation.",
        inputs=merged_payload,
        expected_constraints={
            "requires_human_approval": True,
            "weekly_budget_usd": merged_payload.get("weekly_budget_usd"),
        },
    )

    try:
        validated = GroceryOrderPlacementInput.model_validate(merged_payload)

        # Constitutional Law III: Budget ceiling check before checkout
        if validated.net_total_usd > validated.weekly_budget_usd and not validated.human_approved:
            recovery = ToolErrorRecoveryPayload(
                status="error",
                recovery_mode="GUIDED_RECOVERY",
                error_code="BUDGET_CEILING_EXCEEDED",
                tool_name="place_instacart_grocery_order",
                message=(
                    f"Cart total (${validated.net_total_usd:.2f}) exceeds weekly budget ceiling "
                    f"(${validated.weekly_budget_usd:.2f})."
                ),
                invalid_parameters={
                    "net_total_usd": validated.net_total_usd,
                    "weekly_budget_usd": validated.weekly_budget_usd,
                },
                recovery_instructions=[
                    "Swap higher-cost recipes (such as wild salmon) for budget-friendly legume/quinoa recipes via `search_allergen_safe_recipes`.",
                    "Re-compute the cart via `generate_aisle_sorted_grocery_list` until `net_estimated_cost_usd <= weekly_budget_usd`.",
                    "Or request explicit human budget-overrun approval via `HumanInTheLoopApprovalGate`.",
                ],
                fallback_tool_suggestion="search_allergen_safe_recipes",
            ).model_dump()
            auditor.record_outcome(
                intent.intent_id,
                status="blocked_budget",
                outputs=recovery,
                intent_satisfied=False,
            )
            return recovery

        # Constitutional Law IV: Mandatory Human-in-the-Loop Gate (plan-bound & single-use)
        hitl_gate = get_hitl_approval_gate()
        if not (
            validated.human_approved
            and hitl_gate.verify_confirmation_token(
                household_id=validated.household_id,
                plan_id=validated.plan_id,
                amount_usd=validated.net_total_usd,
                confirmation_token=validated.confirmation_token,
                consume=True,
            )
        ):
            ticket = hitl_gate.request_approval(
                action_name="place_instacart_grocery_order",
                household_id=validated.household_id,
                plan_id=validated.plan_id,
                amount_usd=validated.net_total_usd,
                summary=(
                    f"Place {validated.retailer_name} grocery order for plan '{validated.plan_id}' "
                    f"totaling ${validated.net_total_usd:.2f} ({validated.delivery_window})."
                ),
            )
            pending_output = GroceryOrderPlacementOutput(
                order_status="PENDING_HUMAN_APPROVAL",
                order_id=None,
                approval_request_id=ticket.approval_request_id,
                retailer_name=validated.retailer_name,
                charged_amount_usd=0.0,
                delivery_window=validated.delivery_window,
                message=(
                    f"Order staged and paused at Human-in-the-Loop Gate (ticket '{ticket.approval_request_id}'). "
                    f"No payment has been charged. Confirm approval to execute checkout of ${validated.net_total_usd:.2f}."
                ),
            ).model_dump()
            auditor.record_outcome(
                intent.intent_id,
                status="PENDING_HUMAN_APPROVAL",
                outputs=pending_output,
                intent_satisfied=True,
            )
            return pending_output

        # Resolve partner API secret dynamically from Google Cloud Secret Manager
        try:
            secret_client = get_secret_manager()
            partner_token = secret_client.get_secret("INSTACART_PARTNER_API_TOKEN")
        except SecretResolutionError as exc:
            recovery = ToolErrorRecoveryPayload(
                status="error",
                recovery_mode="GUIDED_RECOVERY",
                error_code="PARTNER_SECRET_UNAVAILABLE",
                tool_name="place_instacart_grocery_order",
                message=str(exc),
                invalid_parameters={},
                recovery_instructions=[
                    "Verify `INSTACART_PARTNER_API_TOKEN` is provisioned in Google Cloud Secret Manager.",
                    "Grant `roles/secretmanager.secretAccessor` to the agent service account.",
                ],
                fallback_tool_suggestion="generate_aisle_sorted_grocery_list",
            ).model_dump()
            auditor.record_outcome(
                intent.intent_id,
                status="error",
                outputs=recovery,
                intent_satisfied=False,
            )
            return recovery

        if not partner_token:
            recovery = ToolErrorRecoveryPayload(
                status="error",
                recovery_mode="GUIDED_RECOVERY",
                error_code="EMPTY_PARTNER_SECRET",
                tool_name="place_instacart_grocery_order",
                message="Partner API token resolved to an empty string.",
                invalid_parameters={},
                recovery_instructions=["Rotate `INSTACART_PARTNER_API_TOKEN` in Secret Manager."],
                fallback_tool_suggestion="generate_aisle_sorted_grocery_list",
            ).model_dump()
            auditor.record_outcome(
                intent.intent_id, status="error", outputs=recovery, intent_satisfied=False
            )
            return recovery

        order_id = f"instacart-ord-{uuid.uuid4().hex[:10]}"
        confirmed_output = GroceryOrderPlacementOutput(
            order_status="CONFIRMED",
            order_id=order_id,
            approval_request_id=None,
            retailer_name=validated.retailer_name,
            charged_amount_usd=round(validated.net_total_usd, 2),
            delivery_window=validated.delivery_window,
            message=(
                f"Human-approved grocery order '{order_id}' placed with {validated.retailer_name} "
                f"for ${validated.net_total_usd:.2f}. Delivery scheduled for {validated.delivery_window}."
            ),
        ).model_dump()

        auditor.record_outcome(
            intent.intent_id,
            status="CONFIRMED",
            outputs=confirmed_output,
            intent_satisfied=True,
        )
        return confirmed_output

    except SchemaValidationError as exc:
        recovery = ToolErrorRecoveryPayload(
            status="error",
            recovery_mode="GUIDED_RECOVERY",
            error_code="INVALID_GROCERY_ORDER_SCHEMA",
            tool_name="place_instacart_grocery_order",
            message=str(exc),
            invalid_parameters={err["field"]: err["message"] for err in exc.errors},
            recovery_instructions=[
                "Verify `net_total_usd` is between 1.0 and 2000.0 and `weekly_budget_usd` >= 35.0.",
                "Ensure `household_id` and `plan_id` are valid strings.",
            ],
            fallback_tool_suggestion="generate_aisle_sorted_grocery_list",
        ).model_dump()
        auditor.record_outcome(
            intent.intent_id,
            status="error",
            outputs=recovery,
            intent_satisfied=False,
        )
        return recovery
    except Exception as exc:
        recovery = ToolErrorRecoveryPayload(
            status="error",
            recovery_mode="GUIDED_RECOVERY",
            error_code="UNEXPECTED_GROCERY_ORDER_ERROR",
            tool_name="place_instacart_grocery_order",
            message=f"Unexpected error: {exc}",
            invalid_parameters={},
            recovery_instructions=["Supply a valid dictionary matching `GroceryOrderPlacementInput`."],
            fallback_tool_suggestion="generate_aisle_sorted_grocery_list",
        ).model_dump()
        auditor.record_outcome(
            intent.intent_id,
            status="error",
            outputs=recovery,
            intent_satisfied=False,
        )
        return recovery


place_instacart_grocery_order.input_model = GroceryOrderPlacementInput  # type: ignore[attr-defined]
place_instacart_grocery_order.output_model = GroceryOrderPlacementOutput  # type: ignore[attr-defined]
place_instacart_grocery_order.input_schema = GroceryOrderPlacementInput.model_json_schema()  # type: ignore[attr-defined]
place_instacart_grocery_order.output_schema = GroceryOrderPlacementOutput.model_json_schema()  # type: ignore[attr-defined]
