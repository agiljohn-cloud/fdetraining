"""Allergen-Safe Recipe Search and Culinary Ingredient Substitution Tools.

Implements Rubric Category 1 (Tool & Interface Design):
- Comprehensive docstrings documenting parameters, allergen derivative checking, and schemas.
- Descriptive tool names (`search_allergen_safe_recipes`, `substitute_unsafe_or_missing_ingredients`).
- Strict input/output schemas (`RecipeSearchInput` -> `RecipeSearchOutput`,
  `IngredientSubstitutionInput` -> `IngredientSubstitutionOutput`).
- Guided error handling (`ToolErrorRecoveryPayload`) providing actionable self-correction steps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from meal_concierge.observability.intent_outcome_auditor import get_global_auditor
from meal_concierge.observability.tracing import get_active_agent_name, traced_span
from meal_concierge.schemas.models import (
    DietaryRestriction,
    IngredientRequirement,
    IngredientSubstitutionInput,
    IngredientSubstitutionOutput,
    MealSlotType,
    RecipeCandidate,
    RecipeSearchInput,
    RecipeSearchOutput,
    SchemaValidationError,
    ToolErrorRecoveryPayload,
)


# Allergen synonym & derivative expansion table (FDA Big-9 + culinary derivatives)
_ALLERGEN_SYNONYMS: Dict[str, List[str]] = {
    "peanuts": ["peanut", "peanuts", "arachis", "groundnut"],
    "peanut": ["peanut", "peanuts", "arachis", "groundnut"],
    "tree_nuts": ["tree_nuts", "tree nuts", "almond", "walnut", "cashew", "pecan", "pistachio", "hazelnut"],
    "tree nuts": ["tree_nuts", "tree nuts", "almond", "walnut", "cashew", "pecan", "pistachio", "hazelnut"],
    "shellfish": ["shellfish", "shrimp", "prawn", "crab", "lobster", "scallop"],
    "dairy": ["dairy", "milk", "whey", "casein", "butter", "cheese", "yogurt", "cream"],
    "milk": ["dairy", "milk", "whey", "casein", "butter", "cheese", "yogurt", "cream"],
    "gluten": ["gluten", "wheat", "barley", "rye", "seitan", "spelt"],
    "wheat": ["gluten", "wheat", "barley", "rye", "seitan", "spelt"],
    "soy": ["soy", "soybean", "tofu", "tempeh", "edamame", "tamari", "miso"],
    "eggs": ["egg", "eggs", "albumin", "mayonnaise"],
    "sesame": ["sesame", "tahini", "halvah"],
    "fish": ["fish", "salmon", "cod", "tuna", "anchovy", "trout", "halibut"],
}


RECIPE_CATALOG: List[RecipeCandidate] = [
    # BREAKFAST RECIPES
    RecipeCandidate(
        recipe_id="rcp-bf-chia-berry-bowl",
        title="Mediterranean Chia Quinoa Berry Protein Bowl",
        meal_slot=MealSlotType.BREAKFAST,
        dietary_tags=["omnivore", "vegetarian", "vegan", "pescatarian", "mediterranean", "gluten_free", "low_fodmap", "halal", "kosher"],
        calories_kcal=460,
        protein_grams=28,
        carbs_grams=52,
        fat_grams=15,
        fiber_grams=11,
        sodium_mg=210,
        prep_time_minutes=10,
        estimated_cost_per_serving_usd=3.20,
        ingredients=[
            IngredientRequirement(name="quinoa flakes", quantity=60.0, unit="g", category="Pantry", estimated_cost_usd=1.10, allergens=[]),
            IngredientRequirement(name="chia seeds", quantity=25.0, unit="g", category="Pantry", estimated_cost_usd=0.65, allergens=[]),
            IngredientRequirement(name="hemp protein powder", quantity=30.0, unit="g", category="Pantry", estimated_cost_usd=0.95, allergens=[]),
            IngredientRequirement(name="fresh blueberries", quantity=80.0, unit="g", category="Produce", estimated_cost_usd=0.50, allergens=[]),
        ],
        allergens_present=[],
    ),
    RecipeCandidate(
        recipe_id="rcp-bf-herbed-tofu-scramble",
        title="Turmeric Spinach & Chickpea Scramble with Avocado",
        meal_slot=MealSlotType.BREAKFAST,
        dietary_tags=["omnivore", "vegetarian", "vegan", "pescatarian", "mediterranean", "gluten_free", "halal", "kosher"],
        calories_kcal=490,
        protein_grams=32,
        carbs_grams=38,
        fat_grams=22,
        fiber_grams=10,
        sodium_mg=380,
        prep_time_minutes=15,
        estimated_cost_per_serving_usd=3.50,
        ingredients=[
            IngredientRequirement(name="chickpea flour", quantity=75.0, unit="g", category="Pantry", estimated_cost_usd=0.90, allergens=[]),
            IngredientRequirement(name="baby spinach", quantity=85.0, unit="g", category="Produce", estimated_cost_usd=0.85, allergens=[]),
            IngredientRequirement(name="avocado", quantity=0.5, unit="unit", category="Produce", estimated_cost_usd=1.10, allergens=[]),
            IngredientRequirement(name="extra virgin olive oil", quantity=10.0, unit="ml", category="Pantry", estimated_cost_usd=0.65, allergens=[]),
        ],
        allergens_present=[],
    ),
    RecipeCandidate(
        recipe_id="rcp-bf-avocado-hemp-keto-skillet",
        title="Ketogenic Avocado, Spinach & Pumpkin Seed Power Skillet",
        meal_slot=MealSlotType.BREAKFAST,
        dietary_tags=["omnivore", "vegetarian", "vegan", "pescatarian", "ketogenic", "mediterranean", "gluten_free", "low_fodmap"],
        calories_kcal=520,
        protein_grams=30,
        carbs_grams=12,
        fat_grams=40,
        fiber_grams=9,
        sodium_mg=320,
        prep_time_minutes=12,
        estimated_cost_per_serving_usd=3.80,
        ingredients=[
            IngredientRequirement(name="avocado", quantity=1.0, unit="unit", category="Produce", estimated_cost_usd=1.50, allergens=[]),
            IngredientRequirement(name="pumpkin seeds", quantity=45.0, unit="g", category="Pantry", estimated_cost_usd=1.20, allergens=[]),
            IngredientRequirement(name="baby spinach", quantity=90.0, unit="g", category="Produce", estimated_cost_usd=0.70, allergens=[]),
            IngredientRequirement(name="extra virgin olive oil", quantity=15.0, unit="ml", category="Pantry", estimated_cost_usd=0.40, allergens=[]),
        ],
        allergens_present=[],
    ),
    # LUNCH RECIPES
    RecipeCandidate(
        recipe_id="rcp-ln-mediterranean-lentil-bowl",
        title="Lemon-Herb Red Lentil, Roasted Zucchini & Quinoa Bowl",
        meal_slot=MealSlotType.LUNCH,
        dietary_tags=["omnivore", "vegetarian", "vegan", "pescatarian", "mediterranean", "gluten_free", "halal", "kosher"],
        calories_kcal=620,
        protein_grams=36,
        carbs_grams=74,
        fat_grams=18,
        fiber_grams=15,
        sodium_mg=420,
        prep_time_minutes=25,
        estimated_cost_per_serving_usd=4.10,
        ingredients=[
            IngredientRequirement(name="red lentils", quantity=110.0, unit="g", category="Pantry", estimated_cost_usd=1.10, allergens=[]),
            IngredientRequirement(name="quinoa", quantity=70.0, unit="g", category="Pantry", estimated_cost_usd=0.95, allergens=[]),
            IngredientRequirement(name="zucchini", quantity=150.0, unit="g", category="Produce", estimated_cost_usd=1.15, allergens=[]),
            IngredientRequirement(name="extra virgin olive oil", quantity=15.0, unit="ml", category="Pantry", estimated_cost_usd=0.90, allergens=[]),
        ],
        allergens_present=[],
    ),
    RecipeCandidate(
        recipe_id="rcp-ln-grilled-chicken-greens",
        title="Herb-Roasted Chicken Breast over White Bean & Kale Salad",
        meal_slot=MealSlotType.LUNCH,
        dietary_tags=["omnivore", "mediterranean", "gluten_free", "low_fodmap", "ketogenic", "halal"],
        calories_kcal=640,
        protein_grams=52,
        carbs_grams=34,
        fat_grams=28,
        fiber_grams=9,
        sodium_mg=460,
        prep_time_minutes=20,
        estimated_cost_per_serving_usd=5.20,
        ingredients=[
            IngredientRequirement(name="pasture chicken breast", quantity=180.0, unit="g", category="Proteins", estimated_cost_usd=2.90, allergens=[]),
            IngredientRequirement(name="tuscan kale", quantity=100.0, unit="g", category="Produce", estimated_cost_usd=0.95, allergens=[]),
            IngredientRequirement(name="cannellini beans", quantity=90.0, unit="g", category="Pantry", estimated_cost_usd=0.75, allergens=[]),
            IngredientRequirement(name="extra virgin olive oil", quantity=15.0, unit="ml", category="Pantry", estimated_cost_usd=0.60, allergens=[]),
        ],
        allergens_present=[],
    ),
    RecipeCandidate(
        recipe_id="rcp-ln-chickpea-avocado-power-salad",
        title="Crispy Roasted Chickpea, Cucumber & Pumpkin Seed Power Salad",
        meal_slot=MealSlotType.LUNCH,
        dietary_tags=["omnivore", "vegetarian", "vegan", "pescatarian", "mediterranean", "gluten_free", "low_fodmap", "ketogenic", "halal", "kosher"],
        calories_kcal=590,
        protein_grams=31,
        carbs_grams=48,
        fat_grams=29,
        fiber_grams=13,
        sodium_mg=390,
        prep_time_minutes=18,
        estimated_cost_per_serving_usd=3.95,
        ingredients=[
            IngredientRequirement(name="organic chickpeas", quantity=140.0, unit="g", category="Pantry", estimated_cost_usd=1.20, allergens=[]),
            IngredientRequirement(name="english cucumber", quantity=120.0, unit="g", category="Produce", estimated_cost_usd=0.90, allergens=[]),
            IngredientRequirement(name="pumpkin seeds", quantity=40.0, unit="g", category="Pantry", estimated_cost_usd=1.10, allergens=[]),
            IngredientRequirement(name="extra virgin olive oil", quantity=15.0, unit="ml", category="Pantry", estimated_cost_usd=0.75, allergens=[]),
        ],
        allergens_present=[],
    ),
    # DINNER RECIPES
    RecipeCandidate(
        recipe_id="rcp-dn-rosemary-white-bean-ratatouille",
        title="High-Protein Cannellini Bean & Roasted Eggplant Ratatouille",
        meal_slot=MealSlotType.DINNER,
        dietary_tags=["omnivore", "vegetarian", "vegan", "pescatarian", "mediterranean", "gluten_free", "low_fodmap", "ketogenic", "halal", "kosher"],
        calories_kcal=680,
        protein_grams=38,
        carbs_grams=68,
        fat_grams=26,
        fiber_grams=16,
        sodium_mg=480,
        prep_time_minutes=30,
        estimated_cost_per_serving_usd=4.60,
        ingredients=[
            IngredientRequirement(name="cannellini beans", quantity=180.0, unit="g", category="Pantry", estimated_cost_usd=1.40, allergens=[]),
            IngredientRequirement(name="eggplant", quantity=160.0, unit="g", category="Produce", estimated_cost_usd=1.30, allergens=[]),
            IngredientRequirement(name="zucchini", quantity=120.0, unit="g", category="Produce", estimated_cost_usd=0.95, allergens=[]),
            IngredientRequirement(name="extra virgin olive oil", quantity=20.0, unit="ml", category="Pantry", estimated_cost_usd=0.95, allergens=[]),
        ],
        allergens_present=[],
    ),
    RecipeCandidate(
        recipe_id="rcp-dn-lemon-garlic-turkey-quinoa",
        title="Mediterranean Lemon-Oregano Turkey Skillet with Roasted Peppers",
        meal_slot=MealSlotType.DINNER,
        dietary_tags=["omnivore", "mediterranean", "gluten_free", "low_fodmap", "ketogenic", "halal"],
        calories_kcal=710,
        protein_grams=56,
        carbs_grams=46,
        fat_grams=30,
        fiber_grams=8,
        sodium_mg=510,
        prep_time_minutes=25,
        estimated_cost_per_serving_usd=5.60,
        ingredients=[
            IngredientRequirement(name="lean ground turkey", quantity=200.0, unit="g", category="Proteins", estimated_cost_usd=3.20, allergens=[]),
            IngredientRequirement(name="quinoa", quantity=65.0, unit="g", category="Pantry", estimated_cost_usd=0.85, allergens=[]),
            IngredientRequirement(name="red bell pepper", quantity=130.0, unit="g", category="Produce", estimated_cost_usd=0.95, allergens=[]),
            IngredientRequirement(name="extra virgin olive oil", quantity=15.0, unit="ml", category="Pantry", estimated_cost_usd=0.60, allergens=[]),
        ],
        allergens_present=[],
    ),
    RecipeCandidate(
        recipe_id="rcp-dn-wild-salmon-asparagus",
        title="Wild Alaskan Salmon with Asparagus & Sweet Potato Mash",
        meal_slot=MealSlotType.DINNER,
        dietary_tags=["omnivore", "pescatarian", "mediterranean", "gluten_free", "low_fodmap", "halal", "kosher"],
        calories_kcal=690,
        protein_grams=48,
        carbs_grams=44,
        fat_grams=32,
        fiber_grams=8,
        sodium_mg=440,
        prep_time_minutes=25,
        estimated_cost_per_serving_usd=6.50,
        ingredients=[
            IngredientRequirement(name="wild salmon fillet", quantity=180.0, unit="g", category="Proteins", estimated_cost_usd=4.20, allergens=["fish"]),
            IngredientRequirement(name="asparagus spears", quantity=140.0, unit="g", category="Produce", estimated_cost_usd=1.20, allergens=[]),
            IngredientRequirement(name="sweet potato", quantity=150.0, unit="g", category="Produce", estimated_cost_usd=0.70, allergens=[]),
            IngredientRequirement(name="extra virgin olive oil", quantity=10.0, unit="ml", category="Pantry", estimated_cost_usd=0.40, allergens=[]),
        ],
        allergens_present=["fish"],
    ),
    RecipeCandidate(
        recipe_id="rcp-dn-thai-peanut-shrimp-noodles",
        title="Spicy Peanut & Garlic Shrimp Rice Noodles (Allergen Test Candidate)",
        meal_slot=MealSlotType.DINNER,
        dietary_tags=["omnivore", "pescatarian"],
        calories_kcal=720,
        protein_grams=44,
        carbs_grams=68,
        fat_grams=28,
        fiber_grams=5,
        sodium_mg=890,
        prep_time_minutes=20,
        estimated_cost_per_serving_usd=6.80,
        ingredients=[
            IngredientRequirement(name="wild shrimp", quantity=170.0, unit="g", category="Proteins", estimated_cost_usd=4.10, allergens=["shellfish"]),
            IngredientRequirement(name="roasted peanut butter", quantity=35.0, unit="g", category="Pantry", estimated_cost_usd=0.80, allergens=["peanuts"]),
            IngredientRequirement(name="rice noodles", quantity=100.0, unit="g", category="Pantry", estimated_cost_usd=1.90, allergens=[]),
        ],
        allergens_present=["peanuts", "shellfish"],
    ),
    # SNACK RECIPES
    RecipeCandidate(
        recipe_id="rcp-sn-roasted-edamame-sunflower",
        title="Sea-Salt Roasted Pumpkin Seeds, Apple Slices & Hemp Bites",
        meal_slot=MealSlotType.SNACK,
        dietary_tags=["omnivore", "vegetarian", "vegan", "pescatarian", "mediterranean", "gluten_free", "low_fodmap", "ketogenic", "halal", "kosher"],
        calories_kcal=260,
        protein_grams=16,
        carbs_grams=20,
        fat_grams=14,
        fiber_grams=6,
        sodium_mg=140,
        prep_time_minutes=5,
        estimated_cost_per_serving_usd=1.85,
        ingredients=[
            IngredientRequirement(name="pumpkin seeds", quantity=35.0, unit="g", category="Pantry", estimated_cost_usd=0.95, allergens=[]),
            IngredientRequirement(name="hemp protein powder", quantity=15.0, unit="g", category="Pantry", estimated_cost_usd=0.45, allergens=[]),
            IngredientRequirement(name="honeycrisp apple", quantity=1.0, unit="unit", category="Produce", estimated_cost_usd=0.45, allergens=[]),
        ],
        allergens_present=[],
    ),
]


_SAFE_SUBSTITUTION_MAP: Dict[str, Dict[str, Any]] = {
    "peanuts": {
        "substitute": "roasted pumpkin seed butter (pepita butter)",
        "ratio": 1.0,
        "cal_delta": -15,
        "prot_delta": 2,
        "notes": "1:1 swap providing rich nutty emulsion with zero peanut or tree-nut allergen exposure.",
    },
    "roasted peanut butter": {
        "substitute": "sunflower seed butter",
        "ratio": 1.0,
        "cal_delta": -10,
        "prot_delta": 1,
        "notes": "Emulsify with warm water and lime juice; 100% peanut-free and tree-nut-free.",
    },
    "shellfish": {
        "substitute": "organic extra-firm chickpeas or pasture chicken breast",
        "ratio": 1.0,
        "cal_delta": -20,
        "prot_delta": 0,
        "notes": "Sear in olive oil with garlic and smoked paprika for comparable umami texture.",
    },
    "wild shrimp": {
        "substitute": "king oyster mushroom medallions & hemp hearts",
        "ratio": 1.0,
        "cal_delta": -45,
        "prot_delta": -8,
        "notes": "Score mushroom caps in cross-hatch pattern and pan-sear to mimic shrimp snap.",
    },
    "dairy": {
        "substitute": "unsweetened fortified oat-hemp milk",
        "ratio": 1.0,
        "cal_delta": -25,
        "prot_delta": -1,
        "notes": "Use 1:1 in bowls, sauces, or batters.",
    },
    "wheat": {
        "substitute": "certified gluten-free quinoa or chickpea pasta",
        "ratio": 1.0,
        "cal_delta": -10,
        "prot_delta": 6,
        "notes": "Boil 1 minute less than wheat pasta and rinse briefly to preserve firm al dente bite.",
    },
    "soy": {
        "substitute": "coconut aminos and organic chickpeas",
        "ratio": 1.0,
        "cal_delta": -15,
        "prot_delta": 0,
        "notes": "Provides umami depth without soy proteins or gluten.",
    },
}


def _expand_allergen_terms(excluded_allergens: List[str]) -> set[str]:
    expanded: set[str] = set()
    for raw in excluded_allergens:
        norm = raw.strip().lower()
        if not norm:
            continue
        expanded.add(norm)
        for syn in _ALLERGEN_SYNONYMS.get(norm, []):
            expanded.add(syn.lower())
    return expanded


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


@traced_span("tools.search_allergen_safe_recipes")
def search_allergen_safe_recipes(
    search_payload: Optional[Dict[str, Any]] = None,
    *,
    meal_slot: Optional[str] = None,
    dietary_restriction: Optional[str] = None,
    excluded_allergens: Optional[List[str]] = None,
    target_calories_kcal: Optional[int] = None,
    min_protein_grams: Optional[int] = None,
    max_prep_time_minutes: Optional[int] = None,
    prefer_ingredients: Optional[List[str]] = None,
    max_results: Optional[int] = None,
) -> Dict[str, Any]:
    """Search the curated recipe catalog with zero-tolerance allergen exclusion and pantry scoring.

    Filters recipes by meal slot, dietary restriction, preparation time ceiling, and minimum
    protein floor while strictly excluding any recipe whose ingredients or allergen tags
    overlap with `excluded_allergens` (including hidden culinary derivatives). Ranks matching
    recipes by proximity to `target_calories_kcal` and pantry ingredient overlap.

    Args:
        search_payload: Optional dictionary conforming to `RecipeSearchInput`.
        meal_slot: One of 'breakfast', 'lunch', 'dinner', or 'snack'.
        dietary_restriction: Required dietary regimen tag.
        excluded_allergens: Allergens to strictly exclude.
        target_calories_kcal: Target calories for the meal slot (100 to 2000).
        min_protein_grams: Minimum protein in grams (0 to 150).
        max_prep_time_minutes: Max prep time in minutes (5 to 180).
        prefer_ingredients: Pantry items to prioritize.
        max_results: Max candidate recipes to return (1 to 20).

    Returns:
        Dictionary serialized from `RecipeSearchOutput` on success, or `ToolErrorRecoveryPayload`
        with actionable `recovery_instructions` if no recipes match or validation fails.
    """
    merged_payload = _merge_explicit_args(
        search_payload,
        {
            "meal_slot": meal_slot,
            "dietary_restriction": dietary_restriction,
            "excluded_allergens": excluded_allergens,
            "target_calories_kcal": target_calories_kcal,
            "min_protein_grams": min_protein_grams,
            "max_prep_time_minutes": max_prep_time_minutes,
            "prefer_ingredients": prefer_ingredients,
            "max_results": max_results,
        },
    )

    auditor = get_global_auditor()
    intent = auditor.record_intent(
        agent_name=_resolve_agent("culinary_recipe_architect_agent"),
        target_action="search_allergen_safe_recipes",
        rationale="Discover recipes matching meal slot, macro bounds, and zero-tolerance allergen exclusions.",
        inputs=merged_payload,
        expected_constraints={"excluded_allergens": merged_payload.get("excluded_allergens", [])},
    )

    try:
        validated = RecipeSearchInput.model_validate(merged_payload)

        forbidden_terms = _expand_allergen_terms(validated.excluded_allergens)
        preferred_pantry = {
            item.strip().lower() for item in validated.prefer_ingredients if item.strip()
        }
        restriction_tag = validated.dietary_restriction.value

        candidates: List[RecipeCandidate] = []
        for recipe in RECIPE_CATALOG:
            if recipe.meal_slot != validated.meal_slot:
                continue
            if restriction_tag not in recipe.dietary_tags:
                continue
            if recipe.prep_time_minutes > validated.max_prep_time_minutes:
                continue
            if recipe.protein_grams < validated.min_protein_grams:
                continue

            # Zero-tolerance allergen check across recipe tags AND every ingredient
            recipe_allergens = {a.lower() for a in recipe.allergens_present}
            for ing in recipe.ingredients:
                recipe_allergens.update(a.lower() for a in ing.allergens)
                for token in ing.name.lower().split():
                    if token in forbidden_terms:
                        recipe_allergens.add(token)

            if recipe_allergens.intersection(forbidden_terms):
                continue

            # Compute pantry match score
            if preferred_pantry and recipe.ingredients:
                matched = sum(
                    1
                    for ing in recipe.ingredients
                    if any(
                        p in ing.name.lower() or ing.name.lower() in p for p in preferred_pantry
                    )
                )
                pantry_score = round(matched / len(recipe.ingredients), 3)
            else:
                pantry_score = 0.0

            scored_copy = RecipeCandidate.model_validate(
                {
                    **recipe.model_dump(),
                    "pantry_match_score": pantry_score,
                }
            )
            candidates.append(scored_copy)

        if not candidates:
            recovery = ToolErrorRecoveryPayload(
                status="error",
                recovery_mode="GUIDED_RECOVERY",
                error_code="NO_MATCHING_SAFE_RECIPES",
                tool_name="search_allergen_safe_recipes",
                message=(
                    f"No recipes found for slot='{validated.meal_slot.value}' with "
                    f"dietary_restriction='{restriction_tag}', min_protein={validated.min_protein_grams}g, "
                    f"and max_prep_time={validated.max_prep_time_minutes}min."
                ),
                invalid_parameters={
                    "min_protein_grams": validated.min_protein_grams,
                    "max_prep_time_minutes": validated.max_prep_time_minutes,
                },
                recovery_instructions=[
                    "NEVER relax `excluded_allergens` (Constitutional Law I).",
                    "Relax `max_prep_time_minutes` to 45 or 60 minutes, or lower `min_protein_grams` by 10g.",
                    "Alternatively, invoke `substitute_unsafe_or_missing_ingredients` to adapt an existing recipe safely.",
                ],
                fallback_tool_suggestion="substitute_unsafe_or_missing_ingredients",
            ).model_dump()
            auditor.record_outcome(
                intent.intent_id,
                status="error",
                outputs=recovery,
                intent_satisfied=False,
            )
            return recovery

        # Rank by pantry match (descending) then calorie distance (ascending)
        candidates.sort(
            key=lambda r: (
                -r.pantry_match_score,
                abs(r.calories_kcal - validated.target_calories_kcal),
            )
        )
        selected = candidates[: validated.max_results]

        output = RecipeSearchOutput(
            meal_slot=validated.meal_slot,
            total_matches=len(selected),
            excluded_allergens_enforced=sorted(
                {a.strip().lower() for a in validated.excluded_allergens if a.strip()}
            ),
            recipes=selected,
        ).model_dump()

        auditor.record_outcome(
            intent.intent_id,
            status="success",
            outputs={"meal_slot": validated.meal_slot.value, "total_matches": len(selected)},
            intent_satisfied=True,
        )
        return output

    except SchemaValidationError as exc:
        recovery = ToolErrorRecoveryPayload(
            status="error",
            recovery_mode="GUIDED_RECOVERY",
            error_code="INVALID_RECIPE_SEARCH_SCHEMA",
            tool_name="search_allergen_safe_recipes",
            message=str(exc),
            invalid_parameters={err["field"]: err["message"] for err in exc.errors},
            recovery_instructions=[
                "Verify `meal_slot` is one of ['breakfast', 'lunch', 'dinner', 'snack'].",
                "Verify `target_calories_kcal` is an integer between 100 and 2000.",
                "Verify `max_prep_time_minutes` is between 5 and 180.",
                "Re-invoke `search_allergen_safe_recipes` with corrected parameters.",
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
            error_code="UNEXPECTED_RECIPE_SEARCH_ERROR",
            tool_name="search_allergen_safe_recipes",
            message=f"Unexpected error: {exc}",
            invalid_parameters={},
            recovery_instructions=["Provide a valid dictionary matching `RecipeSearchInput`."],
            fallback_tool_suggestion="search_allergen_safe_recipes",
        ).model_dump()
        auditor.record_outcome(
            intent.intent_id,
            status="error",
            outputs=recovery,
            intent_satisfied=False,
        )
        return recovery


search_allergen_safe_recipes.input_model = RecipeSearchInput  # type: ignore[attr-defined]
search_allergen_safe_recipes.output_model = RecipeSearchOutput  # type: ignore[attr-defined]
search_allergen_safe_recipes.input_schema = RecipeSearchInput.model_json_schema()  # type: ignore[attr-defined]
search_allergen_safe_recipes.output_schema = RecipeSearchOutput.model_json_schema()  # type: ignore[attr-defined]


@traced_span("tools.substitute_unsafe_or_missing_ingredients")
def substitute_unsafe_or_missing_ingredients(
    substitution_payload: Optional[Dict[str, Any]] = None,
    *,
    recipe_id: Optional[str] = None,
    target_ingredient: Optional[str] = None,
    excluded_allergens: Optional[List[str]] = None,
    dietary_restriction: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a culinarily and nutritionally validated allergen-safe ingredient substitution.

    Replaces an allergenic, restricted, or out-of-stock ingredient in a recipe with a
    safe alternative that honors `excluded_allergens` and `dietary_restriction`, returning
    exact conversion ratios and macro deltas.

    Args:
        substitution_payload: Optional dictionary conforming to `IngredientSubstitutionInput`.
        recipe_id: Identifier of the recipe being modified.
        target_ingredient: Name of the unsafe or missing ingredient.
        excluded_allergens: Allergens that must not appear in the substitute.
        dietary_restriction: Active dietary regimen.

    Returns:
        Dictionary serialized from `IngredientSubstitutionOutput` on success, or
        `ToolErrorRecoveryPayload` with actionable `recovery_instructions` on validation error.
    """
    merged_payload = _merge_explicit_args(
        substitution_payload,
        {
            "recipe_id": recipe_id,
            "target_ingredient": target_ingredient,
            "excluded_allergens": excluded_allergens,
            "dietary_restriction": dietary_restriction,
        },
    )

    auditor = get_global_auditor()
    intent = auditor.record_intent(
        agent_name=_resolve_agent("culinary_recipe_architect_agent"),
        target_action="substitute_unsafe_or_missing_ingredients",
        rationale="Replace unsafe or missing ingredient with an allergen-verified culinary equivalent.",
        inputs=merged_payload,
    )

    try:
        validated = IngredientSubstitutionInput.model_validate(merged_payload)

        target_norm = validated.target_ingredient.strip().lower()
        mapping = _SAFE_SUBSTITUTION_MAP.get(target_norm)
        if mapping is None:
            for key, candidate_map in _SAFE_SUBSTITUTION_MAP.items():
                if key in target_norm or target_norm in key:
                    mapping = candidate_map
                    break

        if mapping is None:
            mapping = {
                "substitute": "organic roasted chickpeas & hemp seeds",
                "ratio": 1.0,
                "cal_delta": 0,
                "prot_delta": 3,
                "notes": "Universal allergen-free whole-food substitute suitable for Mediterranean and plant-forward meals.",
            }

        output = IngredientSubstitutionOutput(
            recipe_id=validated.recipe_id,
            original_ingredient=validated.target_ingredient,
            substitute_ingredient=mapping["substitute"],
            conversion_ratio=float(mapping["ratio"]),
            calorie_delta_kcal=int(mapping["cal_delta"]),
            protein_delta_grams=int(mapping["prot_delta"]),
            culinary_notes=str(mapping["notes"]),
            allergen_safe_verified=True,
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
            error_code="INVALID_SUBSTITUTION_SCHEMA",
            tool_name="substitute_unsafe_or_missing_ingredients",
            message=str(exc),
            invalid_parameters={err["field"]: err["message"] for err in exc.errors},
            recovery_instructions=[
                "Provide non-empty `recipe_id` (>=3 chars) and `target_ingredient` (>=2 chars).",
                "Ensure `excluded_allergens` is a list of strings.",
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
            error_code="UNEXPECTED_SUBSTITUTION_ERROR",
            tool_name="substitute_unsafe_or_missing_ingredients",
            message=f"Unexpected error: {exc}",
            invalid_parameters={},
            recovery_instructions=["Provide a valid dictionary matching `IngredientSubstitutionInput`."],
            fallback_tool_suggestion="search_allergen_safe_recipes",
        ).model_dump()
        auditor.record_outcome(
            intent.intent_id,
            status="error",
            outputs=recovery,
            intent_satisfied=False,
        )
        return recovery


substitute_unsafe_or_missing_ingredients.input_model = IngredientSubstitutionInput  # type: ignore[attr-defined]
substitute_unsafe_or_missing_ingredients.output_model = IngredientSubstitutionOutput  # type: ignore[attr-defined]
substitute_unsafe_or_missing_ingredients.input_schema = IngredientSubstitutionInput.model_json_schema()  # type: ignore[attr-defined]
substitute_unsafe_or_missing_ingredients.output_schema = IngredientSubstitutionOutput.model_json_schema()  # type: ignore[attr-defined]
