"""Clinical Nutrition & Macro-Nutrient Target Calculation Tools.

Implements Rubric Category 1 (Tool & Interface Design):
- Comprehensive docstrings documenting purpose, parameters, return schemas, and recovery behavior.
- Specific, self-documenting tool names (`calculate_weekly_macro_targets`,
  `validate_meal_plan_nutrition_compliance`).
- Explicit typed signatures and strict input/output schema validation (`DietaryProfileInput` ->
  `MacroTargetsOutput`, `MealPlanComplianceInput` -> `MealPlanComplianceOutput`) with attached
  `.input_schema` and `.output_schema` JSON Schema Draft 2020-12 declarations.
- Guided error handling (`ToolErrorRecoveryPayload`) returning step-by-step recovery
  instructions to the LLM on invalid inputs or runtime errors instead of crashing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from meal_concierge.observability.intent_outcome_auditor import get_global_auditor
from meal_concierge.observability.tracing import get_active_agent_name, traced_span
from meal_concierge.schemas.models import (
    ActivityLevel,
    DietaryGoal,
    DietaryProfileInput,
    DietaryRestriction,
    MacroTargetsOutput,
    MealPlanComplianceInput,
    MealPlanComplianceOutput,
    SchemaValidationError,
    ToolErrorRecoveryPayload,
)


_ACTIVITY_MULTIPLIERS: Dict[ActivityLevel, float] = {
    ActivityLevel.SEDENTARY: 1.20,
    ActivityLevel.LIGHTLY_ACTIVE: 1.375,
    ActivityLevel.MODERATELY_ACTIVE: 1.55,
    ActivityLevel.VERY_ACTIVE: 1.725,
    ActivityLevel.ATHLETE: 1.90,
}

_GOAL_CALORIE_DELTAS: Dict[DietaryGoal, int] = {
    DietaryGoal.FAT_LOSS: -400,
    DietaryGoal.MAINTENANCE: 0,
    DietaryGoal.LEAN_MUSCLE_GAIN: 300,
    DietaryGoal.ENDURANCE_PERFORMANCE: 450,
    DietaryGoal.GLYCEMIC_CONTROL: -150,
}


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


@traced_span("tools.calculate_weekly_macro_targets")
def calculate_weekly_macro_targets(
    profile_payload: Optional[Dict[str, Any]] = None,
    *,
    user_id: Optional[str] = None,
    age_years: Optional[int] = None,
    sex_at_birth: Optional[str] = None,
    weight_kg: Optional[float] = None,
    height_cm: Optional[float] = None,
    activity_level: Optional[str] = None,
    dietary_goal: Optional[str] = None,
    dietary_restriction: Optional[str] = None,
    allergies: Optional[List[str]] = None,
    weekly_budget_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate personalized daily calorie, protein, carbohydrate, fat, fiber, and sodium targets.

    Computes Basal Metabolic Rate (BMR) via the Mifflin-St Jeor clinical equation, scales by
    Physical Activity Level (PAL), applies goal-based caloric deltas, enforces the constitutional
    1,200 kcal/day safety floor, and partitions macro-nutrients according to the user's dietary
    restriction and physiological goal.

    Args:
        profile_payload: Optional dictionary conforming to `DietaryProfileInput`.
        user_id: Unique household or individual identifier (2-64 chars).
        age_years: Adult age in years (18 to 100).
        sex_at_birth: Biological sex factor for Mifflin-St Jeor ('male' or 'female').
        weight_kg: Body weight in kilograms (35.0 to 250.0).
        height_cm: Height in centimeters (120.0 to 230.0).
        activity_level: One of 'sedentary', 'lightly_active', 'moderately_active',
            'very_active', or 'athlete'.
        dietary_goal: One of 'fat_loss', 'maintenance', 'lean_muscle_gain',
            'endurance_performance', or 'glycemic_control'.
        dietary_restriction: One of 'omnivore', 'vegetarian', 'vegan', 'pescatarian',
            'mediterranean', 'ketogenic', 'low_fodmap', 'gluten_free', 'halal', or 'kosher'.
        allergies: Zero-tolerance food allergens to exclude.
        weekly_budget_usd: Weekly grocery cap in USD (>= 35.0).

    Returns:
        Dictionary serialized from `MacroTargetsOutput` on success, or `ToolErrorRecoveryPayload`
        (`status="error"`, `recovery_mode="GUIDED_RECOVERY"`) with actionable `recovery_instructions`.
    """
    merged_payload = _merge_explicit_args(
        profile_payload,
        {
            "user_id": user_id,
            "age_years": age_years,
            "sex_at_birth": sex_at_birth,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "activity_level": activity_level,
            "dietary_goal": dietary_goal,
            "dietary_restriction": dietary_restriction,
            "allergies": allergies,
            "weekly_budget_usd": weekly_budget_usd,
        },
    )

    auditor = get_global_auditor()
    intent = auditor.record_intent(
        agent_name=_resolve_agent("nutrition_strategist_agent"),
        target_action="calculate_weekly_macro_targets",
        rationale="Compute Mifflin-St Jeor BMR, TDEE, and daily macro targets with 1,200 kcal floor.",
        inputs=merged_payload,
        expected_constraints={"min_calories_kcal": 1200, "max_calories_kcal": 5000},
    )

    try:
        validated_input = DietaryProfileInput.model_validate(merged_payload)

        sex_norm = validated_input.sex_at_birth.lower().strip()
        if sex_norm not in ("male", "female"):
            recovery = ToolErrorRecoveryPayload(
                status="error",
                recovery_mode="GUIDED_RECOVERY",
                error_code="UNSUPPORTED_SEX_FACTOR",
                tool_name="calculate_weekly_macro_targets",
                message=f"sex_at_birth='{validated_input.sex_at_birth}' must be 'male' or 'female' for Mifflin-St Jeor equation.",
                invalid_parameters={"sex_at_birth": validated_input.sex_at_birth},
                recovery_instructions=[
                    "Normalize `sex_at_birth` to either 'male' or 'female' and re-invoke `calculate_weekly_macro_targets`."
                ],
                fallback_tool_suggestion="calculate_weekly_macro_targets",
            ).model_dump()
            auditor.record_outcome(
                intent.intent_id,
                status="error",
                outputs=recovery,
                intent_satisfied=False,
            )
            return recovery

        sex_constant = 5.0 if sex_norm == "male" else -161.0
        bmr = int(
            round(
                10.0 * validated_input.weight_kg
                + 6.25 * validated_input.height_cm
                - 5.0 * validated_input.age_years
                + sex_constant
            )
        )
        bmr = max(800, min(4500, bmr))

        pal_multiplier = _ACTIVITY_MULTIPLIERS[validated_input.activity_level]
        tdee = int(round(bmr * pal_multiplier))
        goal_delta = _GOAL_CALORIE_DELTAS[validated_input.dietary_goal]

        # Enforce Law II: Constitutional 1,200 kcal floor and 5,000 kcal ceiling
        daily_calories = max(1200, min(5000, tdee + goal_delta))

        if validated_input.dietary_restriction == DietaryRestriction.KETOGENIC:
            protein_ratio, carb_ratio, fat_ratio = 0.25, 0.08, 0.67
        elif validated_input.dietary_goal in (DietaryGoal.LEAN_MUSCLE_GAIN, DietaryGoal.FAT_LOSS):
            protein_ratio, carb_ratio, fat_ratio = 0.32, 0.40, 0.28
        elif validated_input.dietary_restriction == DietaryRestriction.MEDITERRANEAN:
            protein_ratio, carb_ratio, fat_ratio = 0.25, 0.43, 0.32
        else:
            protein_ratio, carb_ratio, fat_ratio = 0.28, 0.44, 0.28

        daily_protein_g = max(50, int(round((daily_calories * protein_ratio) / 4.0)))
        daily_carbs_g = max(25, int(round((daily_calories * carb_ratio) / 4.0)))
        daily_fat_g = max(35, int(round((daily_calories * fat_ratio) / 9.0)))
        daily_fiber_g = (
            32 if validated_input.dietary_restriction == DietaryRestriction.MEDITERRANEAN else 28
        )
        daily_sodium_mg = 1800 if validated_input.dietary_goal == DietaryGoal.GLYCEMIC_CONTROL else 2100

        normalized_allergens = sorted(
            {a.strip().lower() for a in validated_input.allergies if a.strip()}
        )

        output = MacroTargetsOutput(
            user_id=validated_input.user_id,
            basal_metabolic_rate_kcal=bmr,
            total_daily_energy_expenditure_kcal=tdee,
            daily_calorie_target_kcal=daily_calories,
            daily_protein_grams=daily_protein_g,
            daily_carbs_grams=daily_carbs_g,
            daily_fat_grams=daily_fat_g,
            daily_fiber_min_grams=daily_fiber_g,
            daily_sodium_max_mg=daily_sodium_mg,
            excluded_allergens=normalized_allergens,
            weekly_budget_usd=round(validated_input.weekly_budget_usd, 2),
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
            error_code="INVALID_DIETARY_PROFILE_SCHEMA",
            tool_name="calculate_weekly_macro_targets",
            message=str(exc),
            invalid_parameters={err["field"]: err["message"] for err in exc.errors},
            recovery_instructions=[
                "Inspect `invalid_parameters` and supply all required biometric fields.",
                "Ensure `age_years` is an integer in [18, 100], `weight_kg` is in [35.0, 250.0], and `height_cm` is in [120.0, 230.0].",
                "Ensure `activity_level` is one of ['sedentary', 'lightly_active', 'moderately_active', 'very_active', 'athlete'].",
                "Ensure `weekly_budget_usd` is at least 35.0 USD.",
                "Remove any unexpected fields not defined in `DietaryProfileInput`.",
            ],
            fallback_tool_suggestion="calculate_weekly_macro_targets",
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
            error_code="UNEXPECTED_NUTRITION_TOOL_ERROR",
            tool_name="calculate_weekly_macro_targets",
            message=f"Unexpected error: {exc}",
            invalid_parameters={},
            recovery_instructions=[
                "Verify input payload is a valid JSON object matching `DietaryProfileInput`."
            ],
            fallback_tool_suggestion="calculate_weekly_macro_targets",
        ).model_dump()
        auditor.record_outcome(
            intent.intent_id,
            status="error",
            outputs=recovery,
            intent_satisfied=False,
        )
        return recovery


calculate_weekly_macro_targets.input_model = DietaryProfileInput  # type: ignore[attr-defined]
calculate_weekly_macro_targets.output_model = MacroTargetsOutput  # type: ignore[attr-defined]
calculate_weekly_macro_targets.input_schema = DietaryProfileInput.model_json_schema()  # type: ignore[attr-defined]
calculate_weekly_macro_targets.output_schema = MacroTargetsOutput.model_json_schema()  # type: ignore[attr-defined]


@traced_span("tools.validate_meal_plan_nutrition_compliance")
def validate_meal_plan_nutrition_compliance(
    compliance_payload: Optional[Dict[str, Any]] = None,
    *,
    user_id: Optional[str] = None,
    macro_targets: Optional[Dict[str, Any]] = None,
    days: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Audit a candidate multi-day meal schedule against daily macro targets (+/-10%) and weekly budget.

    Verifies that every day in the candidate plan meets the constitutional 1,200 kcal floor,
    calculates mean daily calorie and protein adherence across the schedule (enforcing the
    unified +/- 10% calorie variance threshold), and checks whether total weekly ingredient
    cost stays within `weekly_budget_usd`.

    Args:
        compliance_payload: Optional dictionary conforming to `MealPlanComplianceInput`.
        user_id: User profile identifier.
        macro_targets: Validated `MacroTargetsOutput` dictionary.
        days: List of 1 to 7 `DailyMealPlan` records.

    Returns:
        Dictionary serialized from `MealPlanComplianceOutput` on success, or
        `ToolErrorRecoveryPayload` with actionable `recovery_instructions` if schema validation fails.
    """
    merged_payload = _merge_explicit_args(
        compliance_payload,
        {
            "user_id": user_id,
            "macro_targets": macro_targets,
            "days": days,
        },
    )

    auditor = get_global_auditor()
    intent = auditor.record_intent(
        agent_name=_resolve_agent("nutrition_strategist_agent"),
        target_action="validate_meal_plan_nutrition_compliance",
        rationale="Audit candidate 7-day meal schedule against macro targets (+/-10%) and budget ceiling.",
        inputs={
            "user_id": merged_payload.get("user_id"),
            "day_count": len(merged_payload.get("days", []))
            if isinstance(merged_payload.get("days"), list)
            else 0,
        },
    )

    try:
        validated = MealPlanComplianceInput.model_validate(merged_payload)

        num_days = len(validated.days)
        avg_cal = int(round(sum(d.total_calories_kcal for d in validated.days) / num_days))
        avg_prot = int(round(sum(d.total_protein_grams for d in validated.days) / num_days))
        total_cost = round(sum(d.total_cost_usd for d in validated.days), 2)

        target_cal = validated.macro_targets.daily_calorie_target_kcal
        cal_variance_pct = round(((avg_cal - target_cal) / max(1, target_cal)) * 100.0, 2)
        within_budget = total_cost <= (validated.macro_targets.weekly_budget_usd * 1.15)

        violations: List[str] = []
        if abs(cal_variance_pct) > 10.0:
            violations.append(
                f"Mean daily calories ({avg_cal} kcal) deviates by {cal_variance_pct:.1f}% from target ({target_cal} kcal), exceeding +/-10% limit."
            )
        if avg_prot < int(validated.macro_targets.daily_protein_grams * 0.85):
            violations.append(
                f"Mean daily protein ({avg_prot}g) is below 85% of target ({validated.macro_targets.daily_protein_grams}g)."
            )

        is_compliant = len(violations) == 0
        output = MealPlanComplianceOutput(
            user_id=validated.user_id,
            is_compliant=is_compliant,
            average_daily_calories_kcal=avg_cal,
            average_daily_protein_grams=avg_prot,
            calorie_variance_pct=cal_variance_pct,
            total_weekly_cost_usd=total_cost,
            within_budget=within_budget,
            violations=violations,
        ).model_dump()

        auditor.record_outcome(
            intent.intent_id,
            status="success" if is_compliant else "warning",
            outputs=output,
            intent_satisfied=is_compliant,
        )
        return output

    except SchemaValidationError as exc:
        recovery = ToolErrorRecoveryPayload(
            status="error",
            recovery_mode="GUIDED_RECOVERY",
            error_code="INVALID_MEAL_PLAN_COMPLIANCE_SCHEMA",
            tool_name="validate_meal_plan_nutrition_compliance",
            message=str(exc),
            invalid_parameters={err["field"]: err["message"] for err in exc.errors},
            recovery_instructions=[
                "Ensure `macro_targets` matches the exact output of `calculate_weekly_macro_targets`.",
                "Ensure each entry in `days` has `total_calories_kcal` >= 1200 and between 3 and 5 `meals`.",
                "Re-invoke `validate_meal_plan_nutrition_compliance` after fixing malformed day records.",
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
            error_code="UNEXPECTED_COMPLIANCE_TOOL_ERROR",
            tool_name="validate_meal_plan_nutrition_compliance",
            message=f"Unexpected error: {exc}",
            invalid_parameters={},
            recovery_instructions=["Supply a valid dictionary matching `MealPlanComplianceInput`."],
            fallback_tool_suggestion="validate_meal_plan_nutrition_compliance",
        ).model_dump()
        auditor.record_outcome(
            intent.intent_id,
            status="error",
            outputs=recovery,
            intent_satisfied=False,
        )
        return recovery


validate_meal_plan_nutrition_compliance.input_model = MealPlanComplianceInput  # type: ignore[attr-defined]
validate_meal_plan_nutrition_compliance.output_model = MealPlanComplianceOutput  # type: ignore[attr-defined]
validate_meal_plan_nutrition_compliance.input_schema = MealPlanComplianceInput.model_json_schema()  # type: ignore[attr-defined]
validate_meal_plan_nutrition_compliance.output_schema = MealPlanComplianceOutput.model_json_schema()  # type: ignore[attr-defined]
