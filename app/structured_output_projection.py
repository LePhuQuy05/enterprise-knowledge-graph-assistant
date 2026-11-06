"""Explicit, opt-in projection of unknown top-level NULL metadata only.

The raw reply is immutable evidence. No required field, nested field, claim,
citation, status, limitation or non-null extra can be filled/deleted/repaired.
Strict generation validation still runs after projection; source validation is
the caller's responsibility. This policy does not change historical FAILED
cache entries and must not be reported as native schema compliance.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.generation import (
    MAX_OUTPUT_BYTES,
    GenerationRecipeV9,
    GenerationRecipeV11,
    parse_generated,
)

POLICY = "UNKNOWN_TOP_LEVEL_NULL_EXTRAS_ONLY_V1"


@dataclass(frozen=True)
class ProjectedOutput:
    canonical_output: str
    parsed: object
    raw_sha256: str
    projected_sha256: str
    preserved_fields_sha256: str
    dropped_null_fields: tuple[str, ...]
    policy_sha256: str


def projection_recipe():
    return {"policy": POLICY, "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "maximum_unknown_null_top_level_fields": 4, "nested_projection": False,
        "fill_required_fields": False, "change_known_fields": False, "automatic_retry": False}


def project_null_extras(raw, stop_reason, recipe):
    if not isinstance(raw, str) or not 1 <= len(raw.encode()) <= MAX_OUTPUT_BYTES:
        raise ValueError("OUTPUT_PROJECTION_INPUT_SIZE")
    if not isinstance(recipe, GenerationRecipeV9):
        raise ValueError("OUTPUT_PROJECTION_VERSIONED_SPAN_RECIPE_REQUIRED")
    if isinstance(recipe, GenerationRecipeV11):
        GenerationRecipeV11.model_validate_json(recipe.model_dump_json())
    required = {"reason", "answer_status", "claims"} | ({"limitations"} if isinstance(recipe, GenerationRecipeV11) else set())

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("OUTPUT_PROJECTION_DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("OUTPUT_PROJECTION_NONFINITE_JSON")

    try:
        data = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
        if not isinstance(data, dict):
            raise ValueError("OUTPUT_PROJECTION_OBJECT_REQUIRED")
        if not required <= data.keys():
            raise ValueError("OUTPUT_PROJECTION_MISSING_REQUIRED_FIELDS")
        extras = set(data) - required
        if (len(extras) > 4 or any(data[k] is not None or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", k) for k in extras)):
            raise ValueError("OUTPUT_PROJECTION_NON_NULL_OR_UNBOUNDED_EXTRA")
        preserved = {k: data[k] for k in required}
        serialized = canonical_json(preserved)
        # Identity is checked over JSON values before and after serialization;
        # no model-authored business field is edited to obtain a valid result.
        if canonical_json(json.loads(serialized)) != canonical_json({k: v for k, v in data.items() if k in required}):
            raise ValueError("OUTPUT_PROJECTION_KNOWN_FIELDS_CHANGED")
        parsed = parse_generated(serialized.decode(), stop_reason, recipe)
    except (TypeError, RecursionError, UnicodeError) as exc:
        raise ValueError("OUTPUT_PROJECTION_INVALID_JSON") from exc
    return ProjectedOutput(serialized.decode(), parsed, sha256_bytes(raw.encode()), sha256_bytes(serialized),
        sha256_bytes(canonical_json(preserved)), tuple(sorted(extras)), sha256_bytes(canonical_json(projection_recipe())))
