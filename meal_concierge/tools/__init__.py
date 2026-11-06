"""Domain-Specific Meal Planning Tools with Strict Schemas and Guided Error Recovery."""

from meal_concierge.tools.grocery_order_tools import (
    generate_aisle_sorted_grocery_list,
    place_instacart_grocery_order,
)
from meal_concierge.tools.nutrition_tools import (
    calculate_weekly_macro_targets,
    validate_meal_plan_nutrition_compliance,
)
from meal_concierge.tools.pantry_inventory_tools import (
    check_pantry_inventory_levels,
    reserve_pantry_ingredients_for_plan,
    reset_pantry_inventory_store,
)
from meal_concierge.tools.recipe_search_tools import (
    RECIPE_CATALOG,
    search_allergen_safe_recipes,
    substitute_unsafe_or_missing_ingredients,
)

__all__ = [
    "RECIPE_CATALOG",
    "calculate_weekly_macro_targets",
    "check_pantry_inventory_levels",
    "generate_aisle_sorted_grocery_list",
    "place_instacart_grocery_order",
    "reserve_pantry_ingredients_for_plan",
    "reset_pantry_inventory_store",
    "search_allergen_safe_recipes",
    "substitute_unsafe_or_missing_ingredients",
    "validate_meal_plan_nutrition_compliance",
]
