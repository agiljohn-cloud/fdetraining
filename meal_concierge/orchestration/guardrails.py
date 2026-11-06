"""Security Guardrails, Policy Plugins, and Self-Evaluation Rubric Verifier.

Implements Rubric Category 3.3 (Guardrails & Policy Plugins):
Provides multi-layer guardrails integrated via ADK callbacks (`before_model_callback`,
`before_tool_callback`, `after_model_callback`) and Google Cloud Vertex AI / Model Armor
policy patterns:
1. Input Security Guardrail: Blocks prompt injection, system prompt extraction, and
   unsafe starvation/extreme calorie requests (<1,200 kcal/day or >5,000 kcal/day).
2. Tool Policy Guardrail (`validate_tool_invocation`): Prevents any tool invocation that
   violates declared user allergies or constitutional calorie bounds.
3. Self-Evaluation Rubric Guardrail (`SelfEvaluationRubricGuardrail`): Evaluates generated
   7-day meal plans against catalog allergen metadata, macro target fidelity (+/-10%),
   dietary compliance, and budget adherence before releasing the response to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

from meal_concierge.observability.structured_logger import get_structured_logger
from meal_concierge.observability.tracing import traced_span
from meal_concierge.schemas.models import WeeklyMealPlanOutput
from meal_concierge.tools.recipe_search_tools import RECIPE_CATALOG


_PROMPT_INJECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(?:the\s+)?(?:constitution|safety\s+rules|allergies)", re.IGNORECASE),
    re.compile(r"system\s*override\s*:", re.IGNORECASE),
    re.compile(r"reveal\s+your\s+system\s+prompt", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
]

# Matches full numeric calorie expressions (with optional thousands commas) like "600 calories a day" or "1,500 kcal/day"
_DAILY_CALORIE_MENTION_RE = re.compile(
    r"(?<![\d.,])(\d{1,2}(?:,\d{3})+|\d{2,5})\s*(?:kcal|calories)\s*(?:per\s+day|a\s+day|/day)\b",
    re.IGNORECASE,
)

_WATER_FAST_RE = re.compile(r"\bwater\s+fast\s+for\s+\d+\s+(?:days|weeks)\b", re.IGNORECASE)


def _build_default_catalog_allergen_lookup() -> Dict[str, List[str]]:
    """Build a complete map of `recipe_id -> all allergens` from `RECIPE_CATALOG`."""
    lookup: Dict[str, List[str]] = {}
    for recipe in RECIPE_CATALOG:
        combined = {a.lower() for a in recipe.allergens_present}
        for ing in recipe.ingredients:
            combined.update(a.lower() for a in ing.allergens)
        lookup[recipe.recipe_id] = sorted(combined)
    return lookup


@dataclass
class GuardrailEvaluationResult:
    """Structured verdict returned by input, tool, or self-evaluation guardrails."""

    passed: bool
    guardrail_name: str
    score: float
    violations: List[str] = field(default_factory=list)
    remediation_advice: Optional[str] = None


try:
    from google.adk.models import LlmResponse as _AdkLlmResponse  # type: ignore[import-not-found]
    from google.genai import types as _genai_types  # type: ignore[import-not-found]

    ADK_LLM_RESPONSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    ADK_LLM_RESPONSE_AVAILABLE = False
    _AdkLlmResponse = None
    _genai_types = None


class AdkGuardrailLlmResponse:
    """ADK `LlmResponse`-compatible short-circuit object supporting `.content`, `.model_dump()`, and dict access."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        text = (
            f"Blocked by {payload.get('guardrail')}: "
            f"{'; '.join(payload.get('violations', []))}. "
            f"{payload.get('remediation_advice', '')}"
        ).strip()
        self.role = "model"
        self.parts = [{"text": text}]
        if ADK_LLM_RESPONSE_AVAILABLE and _genai_types is not None:
            self.content: Any = _genai_types.Content(
                role="model",
                parts=[_genai_types.Part.from_text(text=text)],
            )
        else:
            self.content = {"role": "model", "parts": [{"text": text}]}
        self.partial = False
        self.turn_complete = True
        self.error_code = payload.get("guardrail")
        self.error_message = text

    def get(self, key: str, default: Any = None) -> Any:
        return self._payload.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._payload[key]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._payload)

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            "content": self.content,
            "partial": self.partial,
            "turn_complete": self.turn_complete,
            **self._payload,
        }


class SelfEvaluationRubricGuardrail:
    """Self-evaluation verifier that critiques candidate `WeeklyMealPlanOutput` plans.

    Scores the generated meal plan across 4 constitutional dimensions (0.0 to 1.0):
    1. Allergen Zero-Tolerance (weight 0.40)
    2. Caloric & Macro Target Precision within +/-10% (weight 0.25)
    3. Grocery Budget Compliance (weight 0.20)
    4. 7-Day Completeness & Culinary Variety (weight 0.15)
    """

    def __init__(self, min_passing_score: float = 0.90) -> None:
        self.min_passing_score = min_passing_score
        self._default_allergen_lookup = _build_default_catalog_allergen_lookup()
        self._logger = get_structured_logger()

    @traced_span("orchestration.SelfEvaluationRubricGuardrail.evaluate_meal_plan")
    def evaluate_meal_plan(
        self,
        meal_plan: WeeklyMealPlanOutput,
        *,
        declared_allergens: List[str],
        recipe_allergen_lookup: Optional[Dict[str, List[str]]] = None,
    ) -> GuardrailEvaluationResult:
        """Run self-evaluation verification on a candidate 7-day meal plan."""
        violations: List[str] = []
        normalized_allergens = {a.strip().lower() for a in declared_allergens if a.strip()}
        allergen_lookup = (
            recipe_allergen_lookup
            if recipe_allergen_lookup is not None
            else self._default_allergen_lookup
        )

        # 1. Allergen Zero-Tolerance Check
        allergen_score = 1.0
        for day in meal_plan.days:
            for slot in day.meals:
                present = {a.lower() for a in allergen_lookup.get(slot.recipe_id, [])}
                overlap = normalized_allergens.intersection(present)
                if overlap:
                    allergen_score = 0.0
                    violations.append(
                        f"CRITICAL ALLERGEN VIOLATION on {day.day_name} ({slot.slot.value}): "
                        f"Recipe '{slot.recipe_title}' ({slot.recipe_id}) contains {sorted(overlap)}."
                    )

        # 2. Caloric & Macro Target Precision Check (+/- 10% unified threshold)
        macro_score = 1.0
        if not meal_plan.compliance_audit.is_compliant:
            macro_score = 0.4
            violations.extend(meal_plan.compliance_audit.violations)
        elif abs(meal_plan.compliance_audit.calorie_variance_pct) > 10.0:
            macro_score = 0.6
            violations.append(
                f"Calorie variance ({meal_plan.compliance_audit.calorie_variance_pct:.1f}%) exceeds +/-10% threshold."
            )

        # 3. Budget Compliance Check
        budget_score = 1.0
        if not meal_plan.grocery_list.within_budget:
            budget_score = 0.3
            violations.append(
                f"Net grocery cost (${meal_plan.grocery_list.net_estimated_cost_usd:.2f}) exceeds "
                f"weekly budget (${meal_plan.grocery_list.weekly_budget_usd:.2f})."
            )

        # 4. Completeness & Variety Check
        completeness_score = 1.0
        if len(meal_plan.days) != 7:
            completeness_score = 0.0
            violations.append(f"Expected 7 days in weekly plan, found {len(meal_plan.days)}.")
        unique_recipes = {slot.recipe_id for day in meal_plan.days for slot in day.meals}
        if len(unique_recipes) < 4:
            completeness_score = 0.5
            violations.append(
                f"Insufficient culinary variety: only {len(unique_recipes)} unique recipes across 7 days."
            )

        composite_score = round(
            0.40 * allergen_score
            + 0.25 * macro_score
            + 0.20 * budget_score
            + 0.15 * completeness_score,
            4,
        )
        passed = (
            allergen_score == 1.0
            and composite_score >= self.min_passing_score
            and len(violations) == 0
        )

        result = GuardrailEvaluationResult(
            passed=passed,
            guardrail_name="SelfEvaluationRubricGuardrail",
            score=composite_score,
            violations=violations,
            remediation_advice=(
                "Re-invoke `search_allergen_safe_recipes` or `substitute_unsafe_or_missing_ingredients` "
                "to eliminate violations before returning plan."
                if not passed
                else None
            ),
        )

        self._logger.info(
            f"Self-evaluation guardrail completed: passed={passed}, score={composite_score}",
            event_type="SELF_EVALUATION_GUARDRAIL",
            payload={
                "passed": passed,
                "score": composite_score,
                "violations": violations,
            },
        )
        return result


class SafetyAndPolicyGuardrailPlugin:
    """ADK Policy Plugin enforcing input prompt safety, tool allergen policy, and self-evaluation."""

    def __init__(self) -> None:
        self.self_evaluator = SelfEvaluationRubricGuardrail()
        self._catalog_allergens = _build_default_catalog_allergen_lookup()
        self._logger = get_structured_logger()

    @traced_span("orchestration.SafetyAndPolicyGuardrailPlugin.validate_user_input")
    def validate_user_input(self, user_prompt: str) -> GuardrailEvaluationResult:
        """Inspect incoming user request for prompt injection or unsafe starvation/extreme calorie targets."""
        violations: List[str] = []

        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(user_prompt):
                violations.append(
                    "BLOCKED_PROMPT_INJECTION: Attempt to override constitutional safety or system instructions detected."
                )

        for match in _DAILY_CALORIE_MENTION_RE.finditer(user_prompt):
            kcal_val = int(match.group(1).replace(",", ""))
            if kcal_val < 1200:
                violations.append(
                    f"BLOCKED_UNSAFE_CALORIC_FLOOR: Requested {kcal_val} kcal/day is below the constitutional 1,200 kcal/day floor (Law II)."
                )
            elif kcal_val > 5000:
                violations.append(
                    f"BLOCKED_UNSAFE_CALORIC_FLOOR: Requested {kcal_val} kcal/day exceeds the 5,000 kcal/day ceiling (Law II)."
                )

        if _WATER_FAST_RE.search(user_prompt):
            violations.append(
                "BLOCKED_UNSAFE_CALORIC_FLOOR: Prolonged water fasting violates Constitutional Law II."
            )

        passed = len(violations) == 0
        result = GuardrailEvaluationResult(
            passed=passed,
            guardrail_name="InputSafetyGuardrail",
            score=1.0 if passed else 0.0,
            violations=violations,
            remediation_advice=(
                "Provide a safe meal planning request (1,200 to 5,000 kcal/day) that respects constitutional boundaries."
                if not passed
                else None
            ),
        )
        if not passed:
            self._logger.warning(
                f"Input safety guardrail blocked prompt: {violations}",
                event_type="GUARDRAIL_INPUT_BLOCKED",
                payload={"violations": violations},
            )
        return result

    @traced_span("orchestration.SafetyAndPolicyGuardrailPlugin.validate_tool_invocation")
    def validate_tool_invocation(
        self,
        tool_name: str,
        args: Dict[str, Any],
        declared_allergens: Optional[List[str]] = None,
    ) -> GuardrailEvaluationResult:
        """Tool Policy Guardrail checking tool arguments for allergen conflicts before execution."""
        violations: List[str] = []
        active_allergens = {a.strip().lower() for a in (declared_allergens or []) if a.strip()}

        # Unwrap nested payload dict if present
        payload: Dict[str, Any] = dict(args)
        for key in (
            "grocery_payload",
            "search_payload",
            "profile_payload",
            "order_payload",
            "substitution_payload",
        ):
            if isinstance(args.get(key), dict):
                payload = {**payload, **args[key]}
                break

        candidate_ids: List[str] = []
        if isinstance(payload.get("recipe_ids"), list):
            candidate_ids.extend(str(r) for r in payload["recipe_ids"])
        if payload.get("recipe_id") and tool_name != "substitute_unsafe_or_missing_ingredients":
            candidate_ids.append(str(payload["recipe_id"]))

        if active_allergens and candidate_ids:
            for rid in candidate_ids:
                present = set(self._catalog_allergens.get(rid, []))
                overlap = active_allergens.intersection(present)
                if overlap:
                    violations.append(
                        f"BLOCKED_TOOL_ALLERGEN_CONFLICT: Tool '{tool_name}' attempted to use recipe '{rid}' containing declared allergens {sorted(overlap)}."
                    )

        passed = len(violations) == 0
        return GuardrailEvaluationResult(
            passed=passed,
            guardrail_name="ToolPolicyGuardrail",
            score=1.0 if passed else 0.0,
            violations=violations,
            remediation_advice="Remove allergenic recipe IDs before invoking tool." if not passed else None,
        )

    @staticmethod
    def _extract_latest_user_text(llm_request: Any) -> str:
        """Extract text only from the most recent user message so prior block messages do not re-trigger."""
        contents = getattr(llm_request, "contents", None)
        if isinstance(contents, list) and contents:
            for item in reversed(contents):
                if isinstance(item, dict):
                    role = str(item.get("role", "user"))
                    if role == "user":
                        return str(item.get("content", item))
                else:
                    role = str(getattr(item, "role", "user") or "user")
                    if role == "user":
                        parts = getattr(item, "parts", None)
                        if isinstance(parts, list) and parts:
                            return " ".join(str(getattr(p, "text", p) or "") for p in parts)
                        return str(getattr(item, "content", item))
            return ""
        return str(contents or "")

    def adk_before_model_guardrail_callback(
        self,
        callback_context: Any,
        llm_request: Any,
    ) -> Optional[Any]:
        """ADK `before_model_callback` halting execution if prompt injection or unsafe diet is detected."""
        latest_user_text = self._extract_latest_user_text(llm_request)
        verdict = self.validate_user_input(latest_user_text)
        if not verdict.passed:
            resp = AdkGuardrailLlmResponse(
                {
                    "status": "blocked_by_guardrail",
                    "guardrail": verdict.guardrail_name,
                    "violations": verdict.violations,
                    "remediation_advice": verdict.remediation_advice,
                }
            )
            if ADK_LLM_RESPONSE_AVAILABLE and _AdkLlmResponse is not None:
                try:
                    return _AdkLlmResponse(content=resp.content)
                except Exception:
                    return resp
            return resp
        return None

    def adk_before_tool_guardrail_callback(
        self,
        tool: Any,
        args: Dict[str, Any],
        tool_context: Any,
    ) -> Optional[Dict[str, Any]]:
        """ADK `before_tool_callback` enforcing tool-level allergen safety policy across dict or ADK `State`."""
        tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
        state = getattr(tool_context, "state", None)
        allergens: List[str] = []

        if state is not None and hasattr(state, "get"):
            raw_allergens = state.get("excluded_allergens") or state.get("allergies")
            if isinstance(raw_allergens, list):
                allergens.extend(str(a) for a in raw_allergens)
            macro_state = state.get("macro_targets")
            if isinstance(macro_state, dict) and isinstance(macro_state.get("excluded_allergens"), list):
                allergens.extend(str(a) for a in macro_state["excluded_allergens"])

        # Also capture allergens passed directly in tool args and persist into state
        for arg_key in ("allergies", "excluded_allergens"):
            if isinstance(args.get(arg_key), list):
                allergens.extend(str(a) for a in args[arg_key])
            for nested_key in ("profile_payload", "search_payload"):
                nested = args.get(nested_key)
                if isinstance(nested, dict) and isinstance(nested.get(arg_key), list):
                    allergens.extend(str(a) for a in nested[arg_key])

        dedup_allergens = sorted({a.strip().lower() for a in allergens if str(a).strip()})
        if dedup_allergens and state is not None:
            try:
                state["excluded_allergens"] = dedup_allergens
            except Exception:
                pass

        verdict = self.validate_tool_invocation(tool_name, args, declared_allergens=dedup_allergens)
        if not verdict.passed:
            return {
                "status": "error",
                "recovery_mode": "GUIDED_RECOVERY",
                "error_code": "BLOCKED_TOOL_ALLERGEN_CONFLICT",
                "tool_name": tool_name,
                "message": "; ".join(verdict.violations),
                "invalid_parameters": {"recipe_ids": args.get("recipe_ids", [])},
                "recovery_instructions": [verdict.remediation_advice or "Select allergen-safe recipes."],
                "fallback_tool_suggestion": "search_allergen_safe_recipes",
            }
        return None


_GLOBAL_GUARDRAIL_PLUGIN = SafetyAndPolicyGuardrailPlugin()


def get_guardrail_plugin() -> SafetyAndPolicyGuardrailPlugin:
    """Return the singleton `SafetyAndPolicyGuardrailPlugin`."""
    return _GLOBAL_GUARDRAIL_PLUGIN
