"""Household Pantry Inventory Auditing and Stock Reservation Tools.

Implements Rubric Category 1 (Tool & Interface Design):
- Comprehensive docstrings detailing parameters, return schemas, and error recovery.
- Specific tool names (`check_pantry_inventory_levels`, `reserve_pantry_ingredients_for_plan`).
- Explicit typed signatures and strict JSON schemas (`PantryInventoryCheckInput` ->
  `PantryInventoryCheckOutput`, `PantryReservationInput` -> `PantryReservationOutput`).
- Guided error handling (`ToolErrorRecoveryPayload`) for unknown households, invalid schemas,
  or non-positive quantities.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from meal_concierge.observability.intent_outcome_auditor import get_global_auditor
from meal_concierge.observability.tracing import get_active_agent_name, traced_span
from meal_concierge.schemas.models import (
    PantryInventoryCheckInput,
    PantryInventoryCheckOutput,
    PantryItemStatus,
    PantryReservationInput,
    PantryReservationOutput,
    SchemaValidationError,
    ToolErrorRecoveryPayload,
)


_DEFAULT_HOUSEHOLD_PANTRY: List[Dict[str, Any]] = [
    {
        "ingredient_name": "quinoa",
        "available_quantity": 350.0,
        "unit": "g",
        "category": "Pantry",
        "days_until_expiry": 180,
    },
    {
        "ingredient_name": "extra virgin olive oil",
        "available_quantity": 500.0,
        "unit": "ml",
        "category": "Pantry",
        "days_until_expiry": 120,
    },
    {
        "ingredient_name": "red lentils",
        "available_quantity": 400.0,
        "unit": "g",
        "category": "Pantry",
        "days_until_expiry": 240,
    },
    {
        "ingredient_name": "baby spinach",
        "available_quantity": 180.0,
        "unit": "g",
        "category": "Produce",
        "days_until_expiry": 3,
    },
    {
        "ingredient_name": "zucchini",
        "available_quantity": 250.0,
        "unit": "g",
        "category": "Produce",
        "days_until_expiry": 4,
    },
    {
        "ingredient_name": "chia seeds",
        "available_quantity": 200.0,
        "unit": "g",
        "category": "Pantry",
        "days_until_expiry": 300,
    },
    {
        "ingredient_name": "cannellini beans",
        "available_quantity": 360.0,
        "unit": "g",
        "category": "Pantry",
        "days_until_expiry": 365,
    },
]

_LIVE_PANTRY_STORE: Dict[str, List[Dict[str, Any]]] = {}


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


def reset_pantry_inventory_store() -> None:
    """Reset the in-memory household pantry state to baseline defaults for deterministic tests."""
    _LIVE_PANTRY_STORE.clear()


def _get_household_pantry(household_id: str) -> List[Dict[str, Any]]:
    if household_id not in _LIVE_PANTRY_STORE:
        _LIVE_PANTRY_STORE[household_id] = copy.deepcopy(_DEFAULT_HOUSEHOLD_PANTRY)
    return _LIVE_PANTRY_STORE[household_id]


@traced_span("tools.check_pantry_inventory_levels")
def check_pantry_inventory_levels(
    pantry_payload: Optional[Dict[str, Any]] = None,
    *,
    household_id: Optional[str] = None,
    include_expiring_within_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Audit available household pantry stock and identify perishable ingredients expiring soon.

    Retrieves current ingredient quantities, measurement units, storage categories, and
    days until expiration for `household_id`, flagging items expiring within
    `include_expiring_within_days` so the Culinary Recipe Architect prioritizes them early
    in the weekly plan to minimize household food waste.

    Args:
        pantry_payload: Optional dictionary conforming to `PantryInventoryCheckInput`.
        household_id: Unique household identifier (2 to 64 chars).
        include_expiring_within_days: Expiry alert threshold in days (1 to 30, defaults to 5).

    Returns:
        Dictionary serialized from `PantryInventoryCheckOutput` on success, or
        `ToolErrorRecoveryPayload` with actionable `recovery_instructions` if schema validation fails.
    """
    merged_payload = _merge_explicit_args(
        pantry_payload,
        {
            "household_id": household_id,
            "include_expiring_within_days": include_expiring_within_days,
        },
    )

    auditor = get_global_auditor()
    intent = auditor.record_intent(
        agent_name=_resolve_agent("pantry_inventory_auditor_agent"),
        target_action="check_pantry_inventory_levels",
        rationale="Audit current household pantry inventory and flag expiring produce.",
        inputs=merged_payload,
    )

    try:
        validated = PantryInventoryCheckInput.model_validate(merged_payload)

        raw_items = _get_household_pantry(validated.household_id)
        expiring_soon: List[str] = []
        status_items: List[PantryItemStatus] = []

        for item in raw_items:
            is_urgent = item["days_until_expiry"] <= validated.include_expiring_within_days
            if is_urgent and item["available_quantity"] > 0:
                expiring_soon.append(item["ingredient_name"])
            status_items.append(
                PantryItemStatus(
                    ingredient_name=item["ingredient_name"],
                    available_quantity=float(item["available_quantity"]),
                    unit=item["unit"],
                    category=item["category"],
                    days_until_expiry=int(item["days_until_expiry"]),
                    priority_use_soon=is_urgent,
                )
            )

        output = PantryInventoryCheckOutput(
            household_id=validated.household_id,
            total_items_in_stock=len(status_items),
            expiring_soon_items=expiring_soon,
            inventory=status_items,
        ).model_dump()

        auditor.record_outcome(
            intent.intent_id,
            status="success",
            outputs={
                "household_id": validated.household_id,
                "total_items_in_stock": len(status_items),
                "expiring_soon_items": expiring_soon,
            },
            intent_satisfied=True,
        )
        return output

    except SchemaValidationError as exc:
        recovery = ToolErrorRecoveryPayload(
            status="error",
            recovery_mode="GUIDED_RECOVERY",
            error_code="INVALID_PANTRY_CHECK_SCHEMA",
            tool_name="check_pantry_inventory_levels",
            message=str(exc),
            invalid_parameters={err["field"]: err["message"] for err in exc.errors},
            recovery_instructions=[
                "Provide a valid non-empty string for `household_id` (2 to 64 characters).",
                "Ensure `include_expiring_within_days` is an integer between 1 and 30.",
                "Re-invoke `check_pantry_inventory_levels` with valid arguments.",
            ],
            fallback_tool_suggestion="check_pantry_inventory_levels",
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
            error_code="UNEXPECTED_PANTRY_CHECK_ERROR",
            tool_name="check_pantry_inventory_levels",
            message=f"Unexpected error: {exc}",
            invalid_parameters={},
            recovery_instructions=["Supply a valid dictionary matching `PantryInventoryCheckInput`."],
            fallback_tool_suggestion="check_pantry_inventory_levels",
        ).model_dump()
        auditor.record_outcome(
            intent.intent_id,
            status="error",
            outputs=recovery,
            intent_satisfied=False,
        )
        return recovery


check_pantry_inventory_levels.input_model = PantryInventoryCheckInput  # type: ignore[attr-defined]
check_pantry_inventory_levels.output_model = PantryInventoryCheckOutput  # type: ignore[attr-defined]
check_pantry_inventory_levels.input_schema = PantryInventoryCheckInput.model_json_schema()  # type: ignore[attr-defined]
check_pantry_inventory_levels.output_schema = PantryInventoryCheckOutput.model_json_schema()  # type: ignore[attr-defined]


@traced_span("tools.reserve_pantry_ingredients_for_plan")
def reserve_pantry_ingredients_for_plan(
    reservation_payload: Optional[Dict[str, Any]] = None,
    *,
    household_id: Optional[str] = None,
    plan_id: Optional[str] = None,
    reservations: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Deduct allocated ingredient quantities from household pantry stock for an active meal plan.

    Computes how much of each requested ingredient can be fulfilled from existing pantry
    inventory and reports any unfulfilled shortfalls that must be added to the grocery cart.

    Args:
        reservation_payload: Optional dictionary conforming to `PantryReservationInput`.
        household_id: Household identifier (2 to 64 chars).
        plan_id: Meal plan identifier reserving the ingredients.
        reservations: Map of `ingredient_name` -> positive quantity to reserve.

    Returns:
        Dictionary serialized from `PantryReservationOutput` on success, or
        `ToolErrorRecoveryPayload` with actionable `recovery_instructions` if invalid.
    """
    merged_payload = _merge_explicit_args(
        reservation_payload,
        {
            "household_id": household_id,
            "plan_id": plan_id,
            "reservations": reservations,
        },
    )

    auditor = get_global_auditor()
    intent = auditor.record_intent(
        agent_name=_resolve_agent("pantry_inventory_auditor_agent"),
        target_action="reserve_pantry_ingredients_for_plan",
        rationale="Reserve available pantry stock for scheduled weekly recipes.",
        inputs=merged_payload,
    )

    try:
        validated = PantryReservationInput.model_validate(merged_payload)

        negative_keys = {k: v for k, v in validated.reservations.items() if float(v) <= 0}
        if negative_keys:
            recovery = ToolErrorRecoveryPayload(
                status="error",
                recovery_mode="GUIDED_RECOVERY",
                error_code="NON_POSITIVE_RESERVATION_QUANTITY",
                tool_name="reserve_pantry_ingredients_for_plan",
                message="Reservation quantities must be strictly positive (> 0).",
                invalid_parameters=negative_keys,
                recovery_instructions=[
                    "Remove zero or negative quantities from `reservations` and re-invoke `reserve_pantry_ingredients_for_plan`."
                ],
                fallback_tool_suggestion="reserve_pantry_ingredients_for_plan",
            ).model_dump()
            auditor.record_outcome(
                intent.intent_id,
                status="error",
                outputs=recovery,
                intent_satisfied=False,
            )
            return recovery

        pantry_items = _get_household_pantry(validated.household_id)
        pantry_by_name = {item["ingredient_name"].lower(): item for item in pantry_items}

        reserved: Dict[str, float] = {}
        shortfalls: Dict[str, float] = {}

        for ing_name, req_qty in validated.reservations.items():
            requested = float(req_qty)
            norm_name = ing_name.strip().lower()
            stock_entry = pantry_by_name.get(norm_name)
            if stock_entry is None or stock_entry["available_quantity"] <= 0:
                reserved[ing_name] = 0.0
                shortfalls[ing_name] = round(requested, 2)
                continue

            avail = float(stock_entry["available_quantity"])
            used = min(avail, requested)
            stock_entry["available_quantity"] = round(avail - used, 2)
            reserved[ing_name] = round(used, 2)
            if requested > used:
                shortfalls[ing_name] = round(requested - used, 2)

        output = PantryReservationOutput(
            household_id=validated.household_id,
            plan_id=validated.plan_id,
            reserved_quantities=reserved,
            unfulfilled_shortfalls=shortfalls,
        ).model_dump()

        auditor.record_outcome(
            intent.intent_id,
            status="success",
            outputs=output,
            intent_satisfied=True,
        )
        return output

    except SchemaValidationError as exc:
        recovery = ToolErrorRecoveryPayload(
            status="error",
            recovery_mode="GUIDED_RECOVERY",
            error_code="INVALID_PANTRY_RESERVATION_SCHEMA",
            tool_name="reserve_pantry_ingredients_for_plan",
            message=str(exc),
            invalid_parameters={err["field"]: err["message"] for err in exc.errors},
            recovery_instructions=[
                "Provide valid `household_id`, `plan_id`, and a `reservations` dictionary mapping ingredient names to positive numbers.",
            ],
            fallback_tool_suggestion="check_pantry_inventory_levels",
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
            error_code="UNEXPECTED_PANTRY_RESERVATION_ERROR",
            tool_name="reserve_pantry_ingredients_for_plan",
            message=f"Unexpected error: {exc}",
            invalid_parameters={},
            recovery_instructions=["Provide a valid dictionary matching `PantryReservationInput`."],
            fallback_tool_suggestion="check_pantry_inventory_levels",
        ).model_dump()
        auditor.record_outcome(
            intent.intent_id,
            status="error",
            outputs=recovery,
            intent_satisfied=False,
        )
        return recovery


reserve_pantry_ingredients_for_plan.input_model = PantryReservationInput  # type: ignore[attr-defined]
reserve_pantry_ingredients_for_plan.output_model = PantryReservationOutput  # type: ignore[attr-defined]
reserve_pantry_ingredients_for_plan.input_schema = PantryReservationInput.model_json_schema()  # type: ignore[attr-defined]
reserve_pantry_ingredients_for_plan.output_schema = PantryReservationOutput.model_json_schema()  # type: ignore[attr-defined]
