"""AI in 5 Days — Autonomous Meal Planning Concierge Agent package.

Exposes the Google ADK `root_agent` (`meal_concierge_coordinator`) and the
`MealConciergeRunner` orchestration runtime for automated personal meal
planning, clinical macro calculation, allergen-safe recipe discovery, pantry
inventory auditing, and human-in-the-loop grocery procurement.
"""

from meal_concierge.agent import (
    MealConciergeRunner,
    build_meal_concierge_agent_hierarchy,
    meal_concierge_coordinator,
    root_agent,
)

__all__ = [
    "MealConciergeRunner",
    "build_meal_concierge_agent_hierarchy",
    "meal_concierge_coordinator",
    "root_agent",
]

__version__ = "1.0.0"
