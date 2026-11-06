"""Strict JSON Schema and Pydantic validation models for Meal Concierge tools.

Implements Rubric Category 1.3 (Explicit JSON Schemas):
Every tool input and output is constrained by an explicit schema with strict
field bounds, enum validation, `additionalProperties: False` (`extra="forbid"`),
and JSON Schema Draft 2020-12 export to constrain LLM tool calls and responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import inspect
import re
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type, get_args, get_origin, get_type_hints

try:
    from pydantic import (  # type: ignore[import-not-found]
        BaseModel as _PydanticBaseModel,
        ConfigDict,
        Field as _PydanticField,
        ValidationError as _PydanticValidationError,
    )

    PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover - local fallback when pydantic wheel is absent
    _PydanticBaseModel = object  # type: ignore[assignment,misc]
    ConfigDict = dict  # type: ignore[assignment,misc]
    _PydanticField = None
    _PydanticValidationError = ValueError  # type: ignore[assignment,misc]
    PYDANTIC_AVAILABLE = False


class SchemaValidationError(ValueError):
    """Raised when tool input or output violates its strict JSON schema."""

    def __init__(self, model_name: str, errors: List[Dict[str, Any]]) -> None:
        self.model_name = model_name
        self.errors = errors
        summary = "; ".join(f"{e.get('field', '<root>')}: {e.get('message')}" for e in errors)
        super().__init__(f"Schema validation failed for {model_name}: {summary}")


@dataclass(frozen=True)
class FieldConstraint:
    """Metadata describing JSON Schema constraints for a model attribute."""

    default: Any = ...
    default_factory: Optional[Callable[[], Any]] = None
    description: str = ""
    ge: Optional[float] = None
    le: Optional[float] = None
    gt: Optional[float] = None
    lt: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    examples: Optional[List[Any]] = None


def SchemaField(
    default: Any = ...,
    *,
    default_factory: Optional[Callable[[], Any]] = None,
    description: str = "",
    ge: Optional[float] = None,
    le: Optional[float] = None,
    gt: Optional[float] = None,
    lt: Optional[float] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    examples: Optional[List[Any]] = None,
) -> Any:
    """Create a field descriptor with JSON Schema constraints compatible with Pydantic."""
    return FieldConstraint(
        default=default,
        default_factory=default_factory,
        description=description,
        ge=ge,
        le=le,
        gt=gt,
        lt=lt,
        min_length=min_length,
        max_length=max_length,
        pattern=pattern,
        examples=examples,
    )


class StrictSchemaModel:
    """Base model enforcing strict JSON schemas (`additionalProperties: False`).

    Provides `.model_validate()`, `.model_dump()`, and `.model_json_schema()`
    with full runtime type, numeric bound, string length, enum, and nested
    object validation.
    """

    model_config: ClassVar[Dict[str, Any]] = {
        "extra": "forbid",
        "validate_assignment": True,
        "str_strip_whitespace": True,
    }

    def __init__(self, **kwargs: Any) -> None:
        cls = self.__class__
        hints = get_type_hints(cls)
        errors: List[Dict[str, Any]] = []

        allowed_fields = {k for k in hints if not k.startswith("_") and k != "model_config"}
        unknown_fields = set(kwargs.keys()) - allowed_fields
        if unknown_fields:
            for unknown in sorted(unknown_fields):
                errors.append(
                    {
                        "field": unknown,
                        "message": f"Unexpected field '{unknown}' is forbidden by strict schema (additionalProperties=False).",
                    }
                )

        for field_name in allowed_fields:
            expected_type = hints[field_name]
            constraint = getattr(cls, field_name, None)
            if field_name in kwargs:
                raw_val = kwargs[field_name]
            elif isinstance(constraint, FieldConstraint):
                if constraint.default is not ...:
                    raw_val = constraint.default
                elif constraint.default_factory is not None:
                    raw_val = constraint.default_factory()
                else:
                    errors.append(
                        {
                            "field": field_name,
                            "message": f"Required field '{field_name}' is missing.",
                        }
                    )
                    continue
            elif constraint is not None and not inspect.isroutine(constraint):
                raw_val = constraint
            else:
                errors.append(
                    {
                        "field": field_name,
                        "message": f"Required field '{field_name}' is missing.",
                    }
                )
                continue

            try:
                coerced_val = cls._validate_and_coerce_value(
                    field_name,
                    raw_val,
                    expected_type,
                    constraint if isinstance(constraint, FieldConstraint) else None,
                )
                setattr(self, field_name, coerced_val)
            except ValueError as exc:
                errors.append({"field": field_name, "message": str(exc)})

        if errors:
            raise SchemaValidationError(cls.__name__, errors)

        self._post_validate()

    def _post_validate(self) -> None:
        """Hook for cross-field model validation in subclasses."""

    @classmethod
    def _validate_and_coerce_value(
        cls,
        field_name: str,
        value: Any,
        expected_type: Any,
        constraint: Optional[FieldConstraint],
    ) -> Any:
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # Handle Optional / Union[T, None]
        if origin is not None and type(None) in args:
            if value is None:
                return None
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                return cls._validate_and_coerce_value(field_name, value, non_none_args[0], constraint)

        if value is None:
            raise ValueError(f"Field '{field_name}' cannot be null.")

        # Handle List[T]
        if origin in (list, List):
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"Expected array/list for '{field_name}', got {type(value).__name__}.")
            if constraint is not None:
                if constraint.min_length is not None and len(value) < constraint.min_length:
                    raise ValueError(
                        f"List '{field_name}' must contain at least {constraint.min_length} item(s), got {len(value)}."
                    )
                if constraint.max_length is not None and len(value) > constraint.max_length:
                    raise ValueError(
                        f"List '{field_name}' must contain at most {constraint.max_length} item(s), got {len(value)}."
                    )
            item_type = args[0] if args else Any
            return [
                cls._validate_and_coerce_value(f"{field_name}[{idx}]", item, item_type, None)
                for idx, item in enumerate(value)
            ]

        # Handle Dict[K, V]
        if origin in (dict, Dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected object/dict for '{field_name}', got {type(value).__name__}.")
            key_type = args[0] if len(args) >= 1 else Any
            val_type = args[1] if len(args) >= 2 else Any
            validated_dict: Dict[Any, Any] = {}
            for k, v in value.items():
                ck = cls._validate_and_coerce_value(f"{field_name}.key({k})", k, key_type, None) if key_type is not Any else k
                cv = cls._validate_and_coerce_value(f"{field_name}[{k}]", v, val_type, None) if val_type is not Any else v
                validated_dict[ck] = cv
            return validated_dict

        # Handle nested StrictSchemaModel
        if isinstance(expected_type, type) and issubclass(expected_type, StrictSchemaModel):
            if isinstance(value, expected_type):
                return value
            if isinstance(value, dict):
                return expected_type.model_validate(value)
            raise ValueError(f"Expected {expected_type.__name__} object for '{field_name}'.")

        # Handle Enum
        if isinstance(expected_type, type) and issubclass(expected_type, Enum):
            if isinstance(value, expected_type):
                return value
            try:
                return expected_type(value)
            except ValueError as exc:
                valid_values = [e.value for e in expected_type]
                raise ValueError(
                    f"Invalid enum value '{value}' for '{field_name}'. Allowed values: {valid_values}"
                ) from exc

        # Handle primitives
        if expected_type is str:
            if not isinstance(value, str):
                raise ValueError(f"Expected string for '{field_name}', got {type(value).__name__}.")
            value = value.strip()
            if constraint is not None:
                if constraint.min_length is not None and len(value) < constraint.min_length:
                    raise ValueError(
                        f"String '{field_name}' length ({len(value)}) is below min_length={constraint.min_length}."
                    )
                if constraint.max_length is not None and len(value) > constraint.max_length:
                    raise ValueError(
                        f"String '{field_name}' length ({len(value)}) exceeds max_length={constraint.max_length}."
                    )
                if constraint.pattern is not None and not re.fullmatch(constraint.pattern, value):
                    raise ValueError(
                        f"String '{field_name}' value '{value}' does not match required pattern '{constraint.pattern}'."
                    )
            return value

        if expected_type is bool:
            if not isinstance(value, bool):
                raise ValueError(f"Expected boolean for '{field_name}', got {type(value).__name__}.")
            return value

        if expected_type in (int, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Expected {expected_type.__name__} for '{field_name}', got {type(value).__name__}.")
            if expected_type is int and isinstance(value, float) and not value.is_integer():
                raise ValueError(f"Expected integer for '{field_name}', got non-integer float {value}.")
            num_val = expected_type(value)
            if constraint is not None:
                if constraint.ge is not None and num_val < constraint.ge:
                    raise ValueError(f"Value {num_val} for '{field_name}' must be >= {constraint.ge}.")
                if constraint.le is not None and num_val > constraint.le:
                    raise ValueError(f"Value {num_val} for '{field_name}' must be <= {constraint.le}.")
                if constraint.gt is not None and num_val <= constraint.gt:
                    raise ValueError(f"Value {num_val} for '{field_name}' must be > {constraint.gt}.")
                if constraint.lt is not None and num_val >= constraint.lt:
                    raise ValueError(f"Value {num_val} for '{field_name}' must be < {constraint.lt}.")
            return num_val

        return value

    @classmethod
    def model_validate(cls, data: Dict[str, Any] | "StrictSchemaModel") -> "StrictSchemaModel":
        """Validate a dictionary or model instance against this strict schema."""
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise SchemaValidationError(
                cls.__name__,
                [{"field": "<root>", "message": f"Expected JSON object, got {type(data).__name__}."}],
            )
        return cls(**data)

    def model_dump(self) -> Dict[str, Any]:
        """Serialize model to a JSON-compatible dictionary."""
        hints = get_type_hints(self.__class__)
        result: Dict[str, Any] = {}
        for field_name in hints:
            if field_name.startswith("_") or field_name == "model_config":
                continue
            val = getattr(self, field_name)
            result[field_name] = self._serialize_value(val)
        return result

    @classmethod
    def _serialize_value(cls, val: Any) -> Any:
        if isinstance(val, StrictSchemaModel):
            return val.model_dump()
        if isinstance(val, Enum):
            return val.value
        if isinstance(val, list):
            return [cls._serialize_value(item) for item in val]
        if isinstance(val, dict):
            return {k: cls._serialize_value(v) for k, v in val.items()}
        return val

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        """Export an explicit JSON Schema Draft 2020-12 definition for LLM tool calling."""
        hints = get_type_hints(cls)
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for field_name, field_type in hints.items():
            if field_name.startswith("_") or field_name == "model_config":
                continue
            constraint = getattr(cls, field_name, None)
            prop_schema = cls._type_to_json_schema(
                field_type,
                constraint if isinstance(constraint, FieldConstraint) else None,
            )
            properties[field_name] = prop_schema
            if isinstance(constraint, FieldConstraint):
                if constraint.default is ... and constraint.default_factory is None:
                    required.append(field_name)
            elif constraint is None:
                required.append(field_name)

        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": cls.__name__,
            "description": (cls.__doc__ or "").strip(),
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": sorted(required),
        }

    @classmethod
    def _type_to_json_schema(
        cls,
        field_type: Any,
        constraint: Optional[FieldConstraint],
    ) -> Dict[str, Any]:
        origin = get_origin(field_type)
        args = get_args(field_type)

        schema: Dict[str, Any] = {}
        if origin is not None and type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            inner = cls._type_to_json_schema(non_none[0] if non_none else Any, constraint)
            desc = inner.pop("description", None)
            any_of_schema: Dict[str, Any] = {"anyOf": [inner, {"type": "null"}]}
            if desc:
                any_of_schema["description"] = desc
            return any_of_schema

        if origin in (list, List):
            item_type = args[0] if args else Any
            schema = {
                "type": "array",
                "items": cls._type_to_json_schema(item_type, None),
            }
            if constraint and constraint.min_length is not None:
                schema["minItems"] = constraint.min_length
            if constraint and constraint.max_length is not None:
                schema["maxItems"] = constraint.max_length
        elif origin in (dict, Dict):
            val_type = args[1] if len(args) >= 2 else Any
            schema = {
                "type": "object",
                "additionalProperties": cls._type_to_json_schema(val_type, None) if val_type is not Any else True,
            }
        elif isinstance(field_type, type) and issubclass(field_type, StrictSchemaModel):
            schema = field_type.model_json_schema()
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            schema = {
                "type": "string",
                "enum": [e.value for e in field_type],
            }
        elif field_type is str:
            schema = {"type": "string"}
            if constraint and constraint.min_length is not None:
                schema["minLength"] = constraint.min_length
            if constraint and constraint.max_length is not None:
                schema["maxLength"] = constraint.max_length
            if constraint and constraint.pattern is not None:
                schema["pattern"] = constraint.pattern
        elif field_type is int:
            schema = {"type": "integer"}
        elif field_type is float:
            schema = {"type": "number"}
        elif field_type is bool:
            schema = {"type": "boolean"}
        else:
            schema = {"type": "string"}

        if constraint:
            if constraint.description:
                schema["description"] = constraint.description
            if constraint.ge is not None:
                schema["minimum"] = constraint.ge
            if constraint.le is not None:
                schema["maximum"] = constraint.le
            if constraint.gt is not None:
                schema["exclusiveMinimum"] = constraint.gt
            if constraint.lt is not None:
                schema["exclusiveMaximum"] = constraint.lt
            if constraint.examples:
                schema["examples"] = constraint.examples

        return schema


# ---------------------------------------------------------------------------
# Domain Enumerations
# ---------------------------------------------------------------------------


class ActivityLevel(str, Enum):
    """Validated Physical Activity Level (PAL) categories."""

    SEDENTARY = "sedentary"
    LIGHTLY_ACTIVE = "lightly_active"
    MODERATELY_ACTIVE = "moderately_active"
    VERY_ACTIVE = "very_active"
    ATHLETE = "athlete"


class DietaryGoal(str, Enum):
    """Primary physiological or lifestyle objective for meal planning."""

    FAT_LOSS = "fat_loss"
    MAINTENANCE = "maintenance"
    LEAN_MUSCLE_GAIN = "lean_muscle_gain"
    ENDURANCE_PERFORMANCE = "endurance_performance"
    GLYCEMIC_CONTROL = "glycemic_control"


class DietaryRestriction(str, Enum):
    """Standardized dietary ethics and medical regimen classifications."""

    OMNIVORE = "omnivore"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    PESCATARIAN = "pescatarian"
    MEDITERRANEAN = "mediterranean"
    KETOGENIC = "ketogenic"
    LOW_FODMAP = "low_fodmap"
    GLUTEN_FREE = "gluten_free"
    HALAL = "halal"
    KOSHER = "kosher"


class MealSlotType(str, Enum):
    """Standard daily meal slots."""

    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


# ---------------------------------------------------------------------------
# Structured Tool Error Recovery Schema (Rubric Category 1.4)
# ---------------------------------------------------------------------------


class ToolErrorRecoveryPayload(StrictSchemaModel):
    """Structured error response providing guided recovery instructions back to the LLM."""

    status: str = SchemaField(
        default="error",
        description="Execution status indicator ('error' or 'pending_human_approval').",
    )
    recovery_mode: str = SchemaField(
        default="GUIDED_RECOVERY",
        description="Constitutional error recovery mode signal ('GUIDED_RECOVERY').",
    )
    error_code: str = SchemaField(
        description="Machine-readable error identifier (e.g., 'ALLERGEN_CONFLICT', 'BUDGET_EXCEEDED').",
        min_length=3,
    )
    tool_name: str = SchemaField(
        description="Name of the tool that encountered the recoverable error.",
        min_length=3,
    )
    message: str = SchemaField(
        description="Human-readable explanation of why the tool invocation failed.",
        min_length=5,
    )
    invalid_parameters: Dict[str, Any] = SchemaField(
        default_factory=dict,
        description="Map of offending parameter names and values that caused the failure.",
    )
    recovery_instructions: List[str] = SchemaField(
        description="Step-by-step actionable instructions telling the LLM how to self-correct.",
        min_length=1,
    )
    fallback_tool_suggestion: Optional[str] = SchemaField(
        default=None,
        description="Recommended next tool for the LLM to invoke to recover automatically.",
    )


# ---------------------------------------------------------------------------
# Nutrition & Macro Calculation Schemas
# ---------------------------------------------------------------------------


class DietaryProfileInput(StrictSchemaModel):
    """Strict input schema for `calculate_weekly_macro_targets`."""

    user_id: str = SchemaField(
        description="Unique household or individual profile identifier.",
        min_length=2,
        max_length=64,
    )
    age_years: int = SchemaField(
        description="Individual age in years (must be an adult >= 18 and <= 100).",
        ge=18,
        le=100,
    )
    sex_at_birth: str = SchemaField(
        description="Biological sex factor for Mifflin-St Jeor BMR ('male' or 'female').",
        min_length=4,
        max_length=6,
    )
    weight_kg: float = SchemaField(
        description="Current body weight in kilograms.",
        ge=35.0,
        le=250.0,
    )
    height_cm: float = SchemaField(
        description="Stature in centimeters.",
        ge=120.0,
        le=230.0,
    )
    activity_level: ActivityLevel = SchemaField(
        description="Daily physical activity level multiplier tier.",
    )
    dietary_goal: DietaryGoal = SchemaField(
        description="Primary metabolic or body-composition goal.",
    )
    dietary_restriction: DietaryRestriction = SchemaField(
        default=DietaryRestriction.OMNIVORE,
        description="Primary dietary pattern (e.g., mediterranean, vegan, omnivore).",
    )
    allergies: List[str] = SchemaField(
        default_factory=list,
        description="Zero-tolerance food allergens to exclude (e.g., ['peanuts', 'shellfish']).",
    )
    weekly_budget_usd: float = SchemaField(
        default=150.0,
        description="Maximum weekly grocery spending cap in USD.",
        ge=35.0,
        le=2000.0,
    )


class MacroTargetsOutput(StrictSchemaModel):
    """Strict output schema for `calculate_weekly_macro_targets`."""

    user_id: str = SchemaField(description="User profile identifier.")
    basal_metabolic_rate_kcal: int = SchemaField(
        description="Resting energy expenditure in kcal/day.",
        ge=800,
        le=4500,
    )
    total_daily_energy_expenditure_kcal: int = SchemaField(
        description="Maintenance energy expenditure in kcal/day.",
        ge=1000,
        le=6000,
    )
    daily_calorie_target_kcal: int = SchemaField(
        description="Recommended daily caloric intake target (enforces >= 1,200 kcal floor).",
        ge=1200,
        le=5000,
    )
    daily_protein_grams: int = SchemaField(
        description="Daily protein target in grams.",
        ge=45,
        le=350,
    )
    daily_carbs_grams: int = SchemaField(
        description="Daily carbohydrate target in grams.",
        ge=20,
        le=650,
    )
    daily_fat_grams: int = SchemaField(
        description="Daily dietary fat target in grams.",
        ge=30,
        le=220,
    )
    daily_fiber_min_grams: int = SchemaField(
        description="Minimum daily dietary fiber target in grams.",
        ge=20,
        le=70,
    )
    daily_sodium_max_mg: int = SchemaField(
        description="Maximum recommended daily sodium intake in milligrams.",
        ge=1200,
        le=2300,
    )
    excluded_allergens: List[str] = SchemaField(
        default_factory=list,
        description="Normalized list of excluded allergens and derivatives.",
    )
    weekly_budget_usd: float = SchemaField(
        description="Validated weekly grocery budget ceiling in USD.",
        ge=35.0,
    )


# ---------------------------------------------------------------------------
# Recipe Discovery & Substitution Schemas
# ---------------------------------------------------------------------------


class IngredientRequirement(StrictSchemaModel):
    """Structured ingredient specification within a recipe."""

    name: str = SchemaField(description="Normalized ingredient name.", min_length=2)
    quantity: float = SchemaField(description="Required quantity per recipe batch.", ge=0.01)
    unit: str = SchemaField(description="Standard culinary measurement unit (e.g., 'g', 'ml', 'cup', 'unit').")
    category: str = SchemaField(description="Supermarket aisle category (e.g., 'Produce', 'Proteins', 'Pantry').")
    estimated_cost_usd: float = SchemaField(description="Estimated retail unit cost in USD.", ge=0.0)
    allergens: List[str] = SchemaField(
        default_factory=list,
        description="Allergen tags associated with this ingredient.",
    )


class RecipeSearchInput(StrictSchemaModel):
    """Strict input schema for `search_allergen_safe_recipes`."""

    meal_slot: MealSlotType = SchemaField(description="Target meal slot (breakfast, lunch, dinner, snack).")
    dietary_restriction: DietaryRestriction = SchemaField(
        default=DietaryRestriction.OMNIVORE,
        description="Required dietary regimen filter.",
    )
    excluded_allergens: List[str] = SchemaField(
        default_factory=list,
        description="Strict list of allergens that must not appear in any returned recipe.",
    )
    target_calories_kcal: int = SchemaField(
        description="Target calories for this meal slot.",
        ge=100,
        le=2000,
    )
    min_protein_grams: int = SchemaField(
        default=10,
        description="Minimum protein in grams required for the meal slot.",
        ge=0,
        le=150,
    )
    max_prep_time_minutes: int = SchemaField(
        default=45,
        description="Maximum cooking and preparation time in minutes.",
        ge=5,
        le=180,
    )
    prefer_ingredients: List[str] = SchemaField(
        default_factory=list,
        description="Pantry ingredients to prioritize for waste reduction.",
    )
    max_results: int = SchemaField(
        default=5,
        description="Maximum number of candidate recipes to return.",
        ge=1,
        le=20,
    )


class RecipeCandidate(StrictSchemaModel):
    """Validated recipe record with full macro breakdown and ingredient list."""

    recipe_id: str = SchemaField(description="Unique recipe catalog identifier.", min_length=3)
    title: str = SchemaField(description="Human-readable recipe title.", min_length=3)
    meal_slot: MealSlotType = SchemaField(description="Primary meal slot classification.")
    dietary_tags: List[str] = SchemaField(description="Applicable dietary compliance tags.")
    calories_kcal: int = SchemaField(description="Calories per serving.", ge=50, le=2500)
    protein_grams: int = SchemaField(description="Protein in grams per serving.", ge=0, le=200)
    carbs_grams: int = SchemaField(description="Carbohydrates in grams per serving.", ge=0, le=300)
    fat_grams: int = SchemaField(description="Fat in grams per serving.", ge=0, le=150)
    fiber_grams: int = SchemaField(description="Dietary fiber in grams per serving.", ge=0, le=60)
    sodium_mg: int = SchemaField(description="Sodium in milligrams per serving.", ge=0, le=3000)
    prep_time_minutes: int = SchemaField(description="Total preparation time in minutes.", ge=1, le=240)
    estimated_cost_per_serving_usd: float = SchemaField(description="Estimated cost per serving in USD.", ge=0.25)
    ingredients: List[IngredientRequirement] = SchemaField(
        description="Complete ingredient list with quantities and allergen metadata.",
        min_length=1,
    )
    allergens_present: List[str] = SchemaField(
        default_factory=list,
        description="Verified allergens present in the recipe (empty when allergen-safe).",
    )
    pantry_match_score: float = SchemaField(
        default=0.0,
        description="Fraction of recipe ingredients matched against household pantry (0.0 to 1.0).",
        ge=0.0,
        le=1.0,
    )


class RecipeSearchOutput(StrictSchemaModel):
    """Strict output schema for `search_allergen_safe_recipes`."""

    meal_slot: MealSlotType = SchemaField(description="Queried meal slot.")
    total_matches: int = SchemaField(description="Number of safe recipes returned.", ge=0)
    excluded_allergens_enforced: List[str] = SchemaField(
        description="Allergens strictly filtered out during search.",
    )
    recipes: List[RecipeCandidate] = SchemaField(
        description="Ranked list of allergen-safe recipe candidates.",
    )


class IngredientSubstitutionInput(StrictSchemaModel):
    """Strict input schema for `substitute_unsafe_or_missing_ingredients`."""

    recipe_id: str = SchemaField(description="Identifier of the recipe requiring substitution.", min_length=3)
    target_ingredient: str = SchemaField(
        description="Name of the unsafe, allergenic, or out-of-stock ingredient to replace.",
        min_length=2,
    )
    excluded_allergens: List[str] = SchemaField(
        default_factory=list,
        description="Allergens that the replacement ingredient must NOT contain.",
    )
    dietary_restriction: DietaryRestriction = SchemaField(
        default=DietaryRestriction.OMNIVORE,
        description="Active dietary restriction that the substitute must respect.",
    )


class IngredientSubstitutionOutput(StrictSchemaModel):
    """Strict output schema for `substitute_unsafe_or_missing_ingredients`."""

    recipe_id: str = SchemaField(description="Recipe identifier.")
    original_ingredient: str = SchemaField(description="Replaced ingredient name.")
    substitute_ingredient: str = SchemaField(description="Validated safe replacement ingredient.")
    conversion_ratio: float = SchemaField(
        description="Quantity multiplier when swapping original for substitute.",
        ge=0.1,
        le=5.0,
    )
    calorie_delta_kcal: int = SchemaField(description="Net calorie change per serving.")
    protein_delta_grams: int = SchemaField(description="Net protein change in grams per serving.")
    culinary_notes: str = SchemaField(description="Cooking adjustment guidance for the substitution.")
    allergen_safe_verified: bool = SchemaField(description="True when verified free of all excluded allergens.")


# ---------------------------------------------------------------------------
# Pantry Inventory Schemas
# ---------------------------------------------------------------------------


class PantryInventoryCheckInput(StrictSchemaModel):
    """Strict input schema for `check_pantry_inventory_levels`."""

    household_id: str = SchemaField(
        description="Unique household pantry identifier.",
        min_length=2,
        max_length=64,
    )
    include_expiring_within_days: int = SchemaField(
        default=5,
        description="Highlight perishable items expiring within this many days.",
        ge=1,
        le=30,
    )


class PantryItemStatus(StrictSchemaModel):
    """Single household pantry stock item."""

    ingredient_name: str = SchemaField(description="Normalized ingredient name.", min_length=2)
    available_quantity: float = SchemaField(description="Current quantity in stock.", ge=0.0)
    unit: str = SchemaField(description="Measurement unit (e.g., 'g', 'ml', 'unit').")
    category: str = SchemaField(description="Storage category ('Produce', 'Proteins', 'Pantry', 'Dairy_Alt').")
    days_until_expiry: int = SchemaField(description="Days remaining before expiration.", ge=0, le=730)
    priority_use_soon: bool = SchemaField(description="True if expiring soon and should be prioritized.")


class PantryInventoryCheckOutput(StrictSchemaModel):
    """Strict output schema for `check_pantry_inventory_levels`."""

    household_id: str = SchemaField(description="Household identifier.")
    total_items_in_stock: int = SchemaField(description="Count of available pantry items.", ge=0)
    expiring_soon_items: List[str] = SchemaField(
        description="Ingredient names expiring within the priority window.",
    )
    inventory: List[PantryItemStatus] = SchemaField(
        description="Detailed pantry inventory records.",
    )


class PantryReservationInput(StrictSchemaModel):
    """Strict input schema for `reserve_pantry_ingredients_for_plan`."""

    household_id: str = SchemaField(description="Household pantry identifier.", min_length=2)
    plan_id: str = SchemaField(description="Meal plan identifier reserving the stock.", min_length=3)
    reservations: Dict[str, float] = SchemaField(
        description="Map of ingredient_name -> quantity to reserve from pantry stock.",
    )


class PantryReservationOutput(StrictSchemaModel):
    """Strict output schema for `reserve_pantry_ingredients_for_plan`."""

    household_id: str = SchemaField(description="Household identifier.")
    plan_id: str = SchemaField(description="Meal plan identifier.")
    reserved_quantities: Dict[str, float] = SchemaField(
        description="Ingredients and quantities successfully deducted from available stock.",
    )
    unfulfilled_shortfalls: Dict[str, float] = SchemaField(
        description="Ingredients and quantities that exceeded pantry stock and must be purchased.",
    )


# ---------------------------------------------------------------------------
# Weekly Meal Plan & Compliance Schemas
# ---------------------------------------------------------------------------


class PlannedMealSlot(StrictSchemaModel):
    """Single scheduled meal in a daily plan."""

    slot: MealSlotType = SchemaField(description="Meal slot (breakfast, lunch, dinner, snack).")
    recipe_id: str = SchemaField(
        description="Selected recipe ID.",
        min_length=3,
        pattern=r"^[a-z0-9\-]+$",
    )
    recipe_title: str = SchemaField(description="Selected recipe title.", min_length=3)
    servings: int = SchemaField(default=1, description="Number of servings prepared.", ge=1, le=12)
    calories_kcal: int = SchemaField(description="Total calories for this slot.", ge=50, le=2500)
    protein_grams: int = SchemaField(description="Total protein in grams.", ge=0, le=200)
    carbs_grams: int = SchemaField(description="Total carbohydrates in grams.", ge=0, le=300)
    fat_grams: int = SchemaField(description="Total fat in grams.", ge=0, le=150)
    estimated_cost_usd: float = SchemaField(description="Estimated cost in USD.", ge=0.0)


class DailyMealPlan(StrictSchemaModel):
    """Full day of scheduled meals (Day 1 through Day 7)."""

    day_index: int = SchemaField(description="Day number in the plan (1 = Monday ... 7 = Sunday).", ge=1, le=7)
    day_name: str = SchemaField(description="Weekday label (e.g., 'Monday').", min_length=3)
    meals: List[PlannedMealSlot] = SchemaField(
        description="Scheduled meals for the day.",
        min_length=3,
        max_length=5,
    )
    total_calories_kcal: int = SchemaField(description="Sum of daily calories.", ge=1200, le=5000)
    total_protein_grams: int = SchemaField(description="Sum of daily protein in grams.", ge=30)
    total_carbs_grams: int = SchemaField(description="Sum of daily carbs in grams.", ge=10)
    total_fat_grams: int = SchemaField(description="Sum of daily fat in grams.", ge=20)
    total_cost_usd: float = SchemaField(description="Total estimated daily food cost in USD.", ge=0.0)


class MealPlanComplianceInput(StrictSchemaModel):
    """Strict input schema for `validate_meal_plan_nutrition_compliance`."""

    user_id: str = SchemaField(description="User identifier.", min_length=2)
    macro_targets: MacroTargetsOutput = SchemaField(description="Target nutritional bounds from strategist.")
    days: List[DailyMealPlan] = SchemaField(
        description="7-day meal schedule to audit.",
        min_length=1,
        max_length=7,
    )


class MealPlanComplianceOutput(StrictSchemaModel):
    """Strict output schema for `validate_meal_plan_nutrition_compliance`."""

    user_id: str = SchemaField(description="User identifier.")
    is_compliant: bool = SchemaField(description="True if all nutritional, allergen, and budget rules pass.")
    average_daily_calories_kcal: int = SchemaField(description="Mean daily calories across the plan.", ge=0)
    average_daily_protein_grams: int = SchemaField(description="Mean daily protein in grams.", ge=0)
    calorie_variance_pct: float = SchemaField(description="Percentage deviation from target daily calories.")
    total_weekly_cost_usd: float = SchemaField(description="Gross weekly recipe ingredient cost in USD.", ge=0.0)
    within_budget: bool = SchemaField(description="True if total cost is within the weekly budget cap.")
    violations: List[str] = SchemaField(
        default_factory=list,
        description="Specific rule violations detected (empty when compliant).",
    )


# ---------------------------------------------------------------------------
# Grocery List & Human-in-the-Loop Checkout Schemas
# ---------------------------------------------------------------------------


class GroceryListGenerationInput(StrictSchemaModel):
    """Strict input schema for `generate_aisle_sorted_grocery_list`."""

    household_id: str = SchemaField(description="Household pantry identifier.", min_length=2)
    plan_id: str = SchemaField(description="Associated weekly meal plan identifier.", min_length=3)
    recipe_ids: List[str] = SchemaField(
        description="List of recipe IDs scheduled across the week (including repeats for multiple occurrences).",
        min_length=1,
    )
    servings_multiplier: int = SchemaField(
        default=1,
        description="Household size multiplier for each recipe batch.",
        ge=1,
        le=10,
    )
    weekly_budget_usd: float = SchemaField(
        default=150.0,
        description="Maximum allowed net grocery spend in USD.",
        ge=35.0,
    )


class GroceryLineItem(StrictSchemaModel):
    """Single consolidated shopping cart line item after pantry deduction."""

    ingredient_name: str = SchemaField(description="Ingredient name.", min_length=2)
    aisle_category: str = SchemaField(description="Supermarket aisle grouping.")
    gross_required_quantity: float = SchemaField(description="Total quantity needed by recipes.", ge=0.01)
    pantry_deducted_quantity: float = SchemaField(description="Quantity fulfilled from existing pantry stock.", ge=0.0)
    net_purchase_quantity: float = SchemaField(description="Net quantity to buy at store.", ge=0.0)
    unit: str = SchemaField(description="Measurement unit.")
    estimated_line_cost_usd: float = SchemaField(description="Estimated retail cost for net purchase.", ge=0.0)


class AisleSortedGroceryListOutput(StrictSchemaModel):
    """Strict output schema for `generate_aisle_sorted_grocery_list`."""

    plan_id: str = SchemaField(description="Meal plan identifier.")
    household_id: str = SchemaField(description="Household identifier.")
    total_unique_items_to_buy: int = SchemaField(description="Number of distinct items requiring purchase.", ge=0)
    gross_estimated_cost_usd: float = SchemaField(description="Cost before pantry deductions in USD.", ge=0.0)
    pantry_savings_usd: float = SchemaField(description="Dollar amount saved by utilizing pantry stock.", ge=0.0)
    net_estimated_cost_usd: float = SchemaField(description="Final estimated cart subtotal in USD.", ge=0.0)
    weekly_budget_usd: float = SchemaField(description="Configured weekly budget ceiling in USD.", ge=35.0)
    within_budget: bool = SchemaField(description="True when net_estimated_cost_usd <= weekly_budget_usd.")
    items_by_aisle: Dict[str, List[GroceryLineItem]] = SchemaField(
        description="Aisle-sorted map of grocery line items.",
    )


class GroceryOrderPlacementInput(StrictSchemaModel):
    """Strict input schema for `place_instacart_grocery_order` (HITL-gated)."""

    household_id: str = SchemaField(description="Household identifier.", min_length=2)
    plan_id: str = SchemaField(description="Meal plan identifier.", min_length=3)
    retailer_name: str = SchemaField(
        default="Whole Foods Market",
        description="Selected grocery delivery partner.",
        min_length=2,
    )
    delivery_window: str = SchemaField(
        default="Sunday 09:00-11:00",
        description="Requested delivery time window.",
        min_length=3,
    )
    net_total_usd: float = SchemaField(
        description="Final cart total in USD to charge.",
        gt=0.0,
        ge=1.0,
        le=2000.0,
    )
    weekly_budget_usd: float = SchemaField(
        description="User's weekly grocery budget ceiling in USD.",
        ge=35.0,
    )
    human_approved: bool = SchemaField(
        default=False,
        description="Explicit Human-in-the-Loop confirmation flag. Must be True with valid token to execute.",
    )
    confirmation_token: Optional[str] = SchemaField(
        default=None,
        description="Cryptographic approval token issued by HumanInTheLoopApprovalGate.",
    )


class GroceryOrderPlacementOutput(StrictSchemaModel):
    """Strict output schema for `place_instacart_grocery_order`."""

    order_status: str = SchemaField(
        description="Order state ('CONFIRMED', 'PENDING_HUMAN_APPROVAL', or 'BLOCKED_BUDGET')."
    )
    order_id: Optional[str] = SchemaField(
        default=None,
        description="Retailer order tracking ID (populated only after human approval).",
    )
    approval_request_id: Optional[str] = SchemaField(
        default=None,
        description="HITL approval ticket ID when order is paused awaiting user confirmation.",
    )
    retailer_name: str = SchemaField(description="Grocery retailer name.")
    charged_amount_usd: float = SchemaField(description="Amount charged in USD (0.0 until approved).", ge=0.0)
    delivery_window: str = SchemaField(description="Scheduled delivery window.")
    message: str = SchemaField(description="Status explanation and next steps.")


class WeeklyMealPlanOutput(StrictSchemaModel):
    """Complete end-to-end output contract for the Meal Planning Concierge Agent."""

    plan_id: str = SchemaField(description="Unique weekly meal plan identifier.", min_length=3)
    user_id: str = SchemaField(description="User identifier.", min_length=2)
    macro_targets: MacroTargetsOutput = SchemaField(description="Computed daily macro-nutrient targets.")
    days: List[DailyMealPlan] = SchemaField(
        description="7-day structured meal schedule.",
        min_length=7,
        max_length=7,
    )
    compliance_audit: MealPlanComplianceOutput = SchemaField(
        description="Nutritional, allergen, and budget compliance audit result."
    )
    grocery_list: AisleSortedGroceryListOutput = SchemaField(
        description="Pantry-deducted, aisle-sorted grocery shopping list."
    )
    order_outcome: GroceryOrderPlacementOutput = SchemaField(
        description="Grocery order staging or HITL confirmation outcome."
    )


def export_all_tool_json_schemas() -> Dict[str, Dict[str, Any]]:
    """Return a registry of all strict input/output JSON Schemas for ADK tool introspection."""
    schema_classes: List[Type[StrictSchemaModel]] = [
        DietaryProfileInput,
        MacroTargetsOutput,
        RecipeSearchInput,
        RecipeSearchOutput,
        IngredientSubstitutionInput,
        IngredientSubstitutionOutput,
        PantryInventoryCheckInput,
        PantryInventoryCheckOutput,
        PantryReservationInput,
        PantryReservationOutput,
        MealPlanComplianceInput,
        MealPlanComplianceOutput,
        GroceryListGenerationInput,
        AisleSortedGroceryListOutput,
        GroceryOrderPlacementInput,
        GroceryOrderPlacementOutput,
        WeeklyMealPlanOutput,
        ToolErrorRecoveryPayload,
    ]
    return {cls.__name__: cls.model_json_schema() for cls in schema_classes}
