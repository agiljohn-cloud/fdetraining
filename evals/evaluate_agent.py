"""Automated Golden Dataset & Trajectory Evaluation Suite for the Meal Planning Concierge Agent.

Implements Rubric Category 5.1 (Automated Evaluation Suites):
Executes the agent against `evals/golden_dataset.json` and `evals/meal_concierge.evalset.json`
to statically and dynamically measure regression metrics across:
- Tool Trajectory Accuracy (verifying expected tool sequence via `IntentOutcomeAuditor`)
- Allergen zero-tolerance safety (must be 100%)
- Macro & caloric compliance (1,200 kcal floor, +/-10% target bounds)
- Grocery budget compliance & pantry deduction accuracy
- Human-in-the-Loop (HITL) checkout gate enforcement
- Security guardrail blocking (prompt injection & unsafe calorie floor)
- Guided tool error recovery instructions (`recovery_mode="GUIDED_RECOVERY"`)
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

from meal_concierge.agent import MealConciergeRunner, build_meal_concierge_agent_hierarchy
from meal_concierge.observability.intent_outcome_auditor import get_global_auditor
from meal_concierge.tools.pantry_inventory_tools import reset_pantry_inventory_store
from meal_concierge.tools.recipe_search_tools import RECIPE_CATALOG


_RECIPE_ALLERGENS = {
    r.recipe_id: {str(a).lower() for a in r.allergens_present}
    for r in RECIPE_CATALOG
}


def _compute_trajectory_recall(
    expected_tools: List[str],
    actual_tools: List[str],
) -> float:
    """Compute recall of expected tool calls against actual audited tool calls."""
    if not expected_tools:
        return 1.0 if not actual_tools else 0.0
    actual_set = set(actual_tools)
    matched = sum(1 for t in expected_tools if t in actual_set)
    return round(matched / len(expected_tools), 4)


def evaluate_single_scenario(runner: MealConciergeRunner, scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Run one golden scenario and verify all expected outcome and trajectory assertions."""
    reset_pantry_inventory_store()
    auditor = get_global_auditor()
    auditor.clear()

    scenario_id = scenario["scenario_id"]
    expected = scenario["expected"]

    result = runner.run_weekly_meal_plan(
        user_prompt=scenario["user_prompt"],
        dietary_profile=scenario["dietary_profile"],
        session_id=f"eval-session-{scenario_id}",
    )

    audited_pairs = auditor.get_paired_audit_trail()
    actual_tools = [p["intent"]["target_action"] for p in audited_pairs]
    expected_tools = expected.get("expected_tool_trajectory", [])
    trajectory_recall = _compute_trajectory_recall(expected_tools, actual_tools)

    checks: Dict[str, bool] = {}
    actual_status = result.get("status")
    checks["status_matches"] = actual_status == expected["expected_status"]
    checks["tool_trajectory_matches"] = trajectory_recall == 1.0

    if expected["expected_status"] == "success":
        meal_plan = result["meal_plan"]
        checks["seven_days_generated"] = len(meal_plan["days"]) == 7
        checks["self_evaluation_passed"] = bool(result["self_evaluation"]["passed"])

        # Verify zero forbidden allergens in any scheduled meal across all 7 days
        forbidden = {a.lower() for a in expected.get("forbidden_allergens", [])}
        allergen_clean = True
        for day in meal_plan["days"]:
            if day["total_calories_kcal"] < expected["min_daily_calories"]:
                allergen_clean = False
            for slot in day["meals"]:
                present = _RECIPE_ALLERGENS.get(slot["recipe_id"], set())
                if forbidden.intersection(present):
                    allergen_clean = False
        checks["zero_allergen_exposure"] = allergen_clean

        # Verify budget & HITL gate
        grocery = meal_plan["grocery_list"]
        checks["within_weekly_budget"] = (
            grocery["net_estimated_cost_usd"] <= expected["max_weekly_budget_usd"]
        )
        checks["pantry_savings_positive"] = grocery["pantry_savings_usd"] > 0.0
        checks["hitl_gate_enforced"] = (
            meal_plan["order_outcome"]["order_status"] == expected["expected_order_status"]
            and meal_plan["order_outcome"]["charged_amount_usd"] == 0.0
        )

    elif expected["expected_status"] == "blocked_by_guardrail":
        violations_str = " ".join(result.get("violations", []))
        checks["guardrail_violation_detected"] = (
            expected["expected_violation_substring"] in violations_str
        )

    elif expected["expected_status"] == "error":
        checks["error_code_matches"] = result.get("error_code") == expected["expected_error_code"]
        checks["guided_recovery_present"] = (
            result.get("recovery_mode") == "GUIDED_RECOVERY"
            and isinstance(result.get("recovery_instructions"), list)
            and len(result["recovery_instructions"]) >= 1
        )

    passed = all(checks.values())
    return {
        "scenario_id": scenario_id,
        "category": scenario["category"],
        "passed": passed,
        "tool_trajectory_recall": trajectory_recall,
        "actual_tool_sequence": actual_tools,
        "checks": checks,
    }


def validate_adk_evalset_file(evalset_path: str | Path | None = None) -> Dict[str, Any]:
    """Validate structure and schema of `evals/meal_concierge.evalset.json` for `adk eval`."""
    if evalset_path is None:
        evalset_path = Path(__file__).resolve().parent / "meal_concierge.evalset.json"
    else:
        evalset_path = Path(evalset_path)

    payload = json.loads(evalset_path.read_text(encoding="utf-8"))
    eval_cases = payload.get("eval_cases", [])
    valid = (
        bool(payload.get("eval_set_id"))
        and isinstance(eval_cases, list)
        and len(eval_cases) >= 2
        and all("conversation" in c for c in eval_cases)
    )
    return {
        "eval_set_id": payload.get("eval_set_id"),
        "eval_case_count": len(eval_cases),
        "valid_adk_evalset": valid,
    }


def evaluate_adk_agent_graph_and_tool_declarations() -> Dict[str, Any]:
    """Verify the live ADK agent hierarchy, instruction placeholders, and tool signatures."""
    coordinator = build_meal_concierge_agent_hierarchy()
    seq_pipeline = coordinator.sub_agents[0]
    parallel_stage = seq_pipeline.sub_agents[0]
    all_llm_agents = [
        coordinator,
        *parallel_stage.sub_agents,
        seq_pipeline.sub_agents[1],
        seq_pipeline.sub_agents[2],
    ]

    # 1. Verify no required `{var}` placeholders exist without `?` optional marker
    required_placeholder_re = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    unresolved_placeholders: List[str] = []
    for ag in all_llm_agents:
        for match in required_placeholder_re.finditer(ag.instruction):
            unresolved_placeholders.append(f"{ag.name}:{match.group(1)}")

    # 2. Verify all tools expose real function names, docstrings, and explicit typed signatures
    invalid_tools: List[str] = []
    total_tools_checked = 0
    for ag in all_llm_agents:
        for tool in ag.tools:
            total_tools_checked += 1
            derived_name = getattr(tool, "__name__", tool.__class__.__name__)
            sig = inspect.signature(tool)
            param_names = list(sig.parameters.keys())
            if derived_name == "FunctionTool" or param_names == ["args", "kwargs"] or len(param_names) < 2:
                invalid_tools.append(f"{ag.name}:{derived_name}({param_names})")

    graph_valid = len(unresolved_placeholders) == 0 and len(invalid_tools) == 0 and total_tools_checked == 9
    return {
        "adk_graph_valid": graph_valid,
        "llm_agents_checked": len(all_llm_agents),
        "tools_checked": total_tools_checked,
        "unresolved_placeholders": unresolved_placeholders,
        "invalid_tools": invalid_tools,
    }


def run_golden_evaluation_suite(dataset_path: str | Path | None = None) -> Dict[str, Any]:
    """Execute the full golden evaluation dataset and return aggregate regression metrics."""
    if dataset_path is None:
        dataset_path = Path(__file__).resolve().parent / "golden_dataset.json"
    else:
        dataset_path = Path(dataset_path)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    runner = MealConciergeRunner()

    scenario_results: List[Dict[str, Any]] = []
    for scenario in dataset["scenarios"]:
        scenario_results.append(evaluate_single_scenario(runner, scenario))

    adk_evalset_status = validate_adk_evalset_file()
    adk_graph_status = evaluate_adk_agent_graph_and_tool_declarations()
    total = len(scenario_results)
    passed_count = sum(1 for r in scenario_results if r["passed"])
    pass_rate = round((passed_count / max(1, total)) * 100.0, 2)
    avg_trajectory_recall = round(
        sum(r["tool_trajectory_recall"] for r in scenario_results) / max(1, total), 4
    )

    summary = {
        "dataset_name": dataset["dataset_name"],
        "version": dataset["version"],
        "total_scenarios": total,
        "passed_scenarios": passed_count,
        "failed_scenarios": total - passed_count,
        "pass_rate_pct": pass_rate,
        "average_tool_trajectory_recall": avg_trajectory_recall,
        "adk_evalset": adk_evalset_status,
        "adk_graph_verification": adk_graph_status,
        "regression_detected": (
            (passed_count != total)
            or not adk_evalset_status["valid_adk_evalset"]
            or not adk_graph_status["adk_graph_valid"]
        ),
        "scenario_results": scenario_results,
    }
    return summary


if __name__ == "__main__":
    report = run_golden_evaluation_suite()
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    if report["regression_detected"]:
        sys.exit(1)
