"""
metrics_params.py

Phase 2C — Parameter validation metrics.

For each step in a generated workflow, extracts the tool's parameter schema
from that sample's own system message (via context_tool_parser) and validates
the generated params against it.

Checks per step:
  1. required_params_present — all required params are in step.params
  2. no_extra_params         — no params outside the known param list
  3. param_types_correct     — all present params have correct types
                               (only when type info is available in the schema)

Samples whose system message has no extractable schema are skipped and
reported via param_schema_coverage_rate so the gap is visible.

JSON Schema type → Python type mapping:
  "string"  → str
  "number"  → int | float
  "integer" → int
  "boolean" → bool
  "array"   → list
  "object"  → dict

All rates are floats in [0.0, 1.0].
"""

from __future__ import annotations

import json
from loguru import logger

from model_2_training.src.eval.context_tool_parser import get_sample_tool_schemas


# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------

_JSON_SCHEMA_TYPE_MAP: dict[str, type | tuple] = {
    "string":  str,
    "number":  (int, float),
    "integer": int,
    "boolean": bool,
    "array":   list,
    "object":  dict,
}


def _check_type(value: object, type_str: str) -> bool:
    """Return True if value matches the JSON Schema type string."""
    expected = _JSON_SCHEMA_TYPE_MAP.get(type_str.lower())
    if expected is None:
        return True  # unknown type — don't penalise
    # booleans are a subtype of int in Python; handle explicitly
    if type_str == "integer" and isinstance(value, bool):
        return False
    if type_str == "number" and isinstance(value, bool):
        return False
    return isinstance(value, expected)


# ---------------------------------------------------------------------------
# Per-step validation
# ---------------------------------------------------------------------------

def validate_step_params(step: dict, tool_schema: dict) -> dict:
    """
    Validate a single step's params against its tool schema.

    Args:
        step:        A step dict with "tool" and "params" keys.
        tool_schema: Normalised schema dict from context_tool_parser
                     with keys: all_params, required_params, param_types, has_schema.

    Returns:
        {
          "required_params_present": bool,
          "no_extra_params":         bool,
          "param_types_correct":     bool | None,  # None = type info unavailable
          "missing_params":          list[str],
          "extra_params":            list[str],
          "type_errors":             dict[str, str],  # {param: "expected X got Y"}
        }
    """
    params = step.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    all_params     = set(tool_schema.get("all_params", []))
    required_params = set(tool_schema.get("required_params", []))
    param_types    = tool_schema.get("param_types", {})
    given_params   = set(params.keys())

    # Check 1: required params present
    missing = sorted(required_params - given_params)

    # Check 2: no extra params (only when all_params is non-empty)
    if all_params:
        extra = sorted(given_params - all_params)
    else:
        extra = []  # can't determine extras if schema has no param list

    # Check 3: type correctness (only when param_types available)
    type_errors: dict[str, str] = {}
    if param_types:
        for param, value in params.items():
            if param in param_types:
                expected_type = param_types[param]
                if not _check_type(value, expected_type):
                    type_errors[param] = (
                        f"expected {expected_type} got {type(value).__name__}"
                    )

    types_checked = bool(param_types)

    return {
        "required_params_present": len(missing) == 0,
        "no_extra_params":         len(extra) == 0,
        "param_types_correct":     (len(type_errors) == 0) if types_checked else None,
        "missing_params":          missing,
        "extra_params":            extra,
        "type_errors":             type_errors,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_param_metrics(predictions: list[dict]) -> dict[str, float | int]:
    """
    Compute Phase 2C parameter validation metrics over a list of predictions.

    For each prediction:
      1. Extracts tool schemas from the sample's system message.
      2. For each step with a known tool schema, validates params.
      3. Steps without an extractable schema are counted in coverage but skipped.

    Args:
        predictions: List of prediction dicts with "generated", "reference",
                     and "sample" keys (as produced by generator.py).

    Returns:
        Dict with rates (floats in [0.0, 1.0]) and denominator counts.
    """
    if not predictions:
        logger.warning("compute_param_metrics called with empty predictions list.")
        return {}

    # Coverage tracking (all tool-workflow steps)
    steps_total       = 0   # all steps across all predictions
    steps_with_schema = 0   # steps where we had a schema to validate against

    # Validation counters (only over steps_with_schema)
    required_pass  = 0
    no_extras_pass = 0
    types_pass     = 0
    types_total    = 0  # steps where type info was available
    full_pass      = 0  # steps passing all available checks

    for pred in predictions:
        generated = pred.get("generated", "") or ""

        # Parse generated JSON
        try:
            data = json.loads(generated.strip())
        except (json.JSONDecodeError, ValueError):
            continue

        if not isinstance(data, dict):
            continue

        workflow = data.get("workflow", {})
        if not isinstance(workflow, dict):
            continue

        steps = workflow.get("steps", [])
        if not isinstance(steps, list) or not steps:
            continue  # message-shape or no steps

        # Extract per-sample tool schemas from system message
        tool_schemas = get_sample_tool_schemas(pred)

        for step in steps:
            if not isinstance(step, dict):
                continue

            tool_name = step.get("tool", "")
            steps_total += 1

            schema = tool_schemas.get(tool_name)
            if schema is None or not schema.get("has_schema"):
                continue  # no schema available — skip validation, track coverage gap

            steps_with_schema += 1
            result = validate_step_params(step, schema)

            req_ok   = result["required_params_present"]
            extra_ok = result["no_extra_params"]
            type_ok  = result["param_types_correct"]  # bool or None

            required_pass  += int(req_ok)
            no_extras_pass += int(extra_ok)

            if type_ok is not None:
                types_total += 1
                types_pass  += int(type_ok)

            # full_pass: must pass all checks that were computable
            all_checks_pass = req_ok and extra_ok and (type_ok if type_ok is not None else True)
            full_pass += int(all_checks_pass)

    def _rate(num: int, denom: int) -> float:
        return round(num / denom, 4) if denom > 0 else 0.0

    metrics: dict[str, float | int] = {
        # Coverage
        "param_schema_coverage_rate": _rate(steps_with_schema, steps_total),
        "param_schema_coverage_denominator": steps_total,

        # Validation (over steps with schema)
        "param_completeness_rate":     _rate(required_pass,  steps_with_schema),
        "param_no_extras_rate":        _rate(no_extras_pass, steps_with_schema),
        "param_type_correctness_rate": _rate(types_pass,     types_total),
        "param_type_correctness_denominator": types_total,
        "full_param_validity_rate":    _rate(full_pass,      steps_with_schema),
        "param_validated_steps":       steps_with_schema,
    }

    logger.info("Parameter Validation Metrics (Phase 2C):")
    logger.info(f"  {'param_schema_coverage_rate':<35} {metrics['param_schema_coverage_rate']}  "
                f"({steps_with_schema}/{steps_total} steps had a schema)")
    rate_keys = [k for k in metrics if k.endswith("_rate") and k != "param_schema_coverage_rate"]
    for k in rate_keys:
        denom_key = k.replace("_rate", "_denominator")
        denom = metrics.get(denom_key, steps_with_schema)
        logger.info(f"  {k:<35} {metrics[k]}  (n={denom})")

    return metrics


def summarize_param_errors(predictions: list[dict], n_samples: int = 10) -> list[dict]:
    """
    Return a sample of steps that failed parameter validation, for manual inspection.

    Returns list of dicts with tool, params, schema, and validation result.
    """
    errors = []
    for pred in predictions:
        generated = pred.get("generated", "") or ""
        try:
            data = json.loads(generated.strip())
        except (json.JSONDecodeError, ValueError):
            continue

        if not isinstance(data, dict):
            continue

        steps = (data.get("workflow") or {}).get("steps", [])
        if not isinstance(steps, list):
            continue

        tool_schemas = get_sample_tool_schemas(pred)

        for step in steps:
            if not isinstance(step, dict):
                continue

            tool_name = step.get("tool", "")
            schema = tool_schemas.get(tool_name)
            if schema is None or not schema.get("has_schema"):
                continue

            result = validate_step_params(step, schema)
            if not result["required_params_present"] or not result["no_extra_params"] or result["type_errors"]:
                errors.append({
                    "tool":      tool_name,
                    "params":    step.get("params"),
                    "schema":    schema,
                    "missing":   result["missing_params"],
                    "extra":     result["extra_params"],
                    "type_errors": result["type_errors"],
                })
            if len(errors) >= n_samples:
                return errors

    return errors
