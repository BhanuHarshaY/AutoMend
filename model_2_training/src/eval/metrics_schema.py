"""
metrics_schema.py

Phase 2A — Pydantic schema validation metrics.

Runs full Pydantic validation on every generated output and measures how many
conform to the expected WorkflowResponse schema. Builds on top of Phase 1
(structural metrics) by catching failures that json.loads() misses:
  - Missing "workflow" root key
  - "steps" not a list
  - Steps missing "tool" or "params" fields
  - Wrong field types at any level
  - Unexpected extra keys
  - Empty tool names
  - Message present when steps are non-empty (or absent when steps are empty)

All rates are floats in [0.0, 1.0].
Predictions that failed JSON parsing in Phase 1 are counted as schema failures.
"""

from __future__ import annotations

import json
from collections import Counter
from loguru import logger

from model_2_training.src.schemas.workflow_schema import (
    OutputShape,
    infer_shape,
    parse_response,
)


# ---------------------------------------------------------------------------
# Error type constants (used in schema_error_distribution)
# ---------------------------------------------------------------------------

ERR_EMPTY = "empty_output"
ERR_JSON_PARSE = "json_parse_failed"
ERR_NOT_DICT = "not_a_dict"
ERR_SCHEMA = "schema_invalid"
ERR_EXTRA_FIELDS = "extra_forbidden"
ERR_WRONG_TYPE = "wrong_type"
ERR_MISSING_FIELD = "missing"


def _classify_pydantic_errors(text: str) -> list[str]:
    """
    Run Pydantic validation and return a list of error type strings.
    Empty list means the output is schema-valid.
    """
    if not text or not text.strip():
        return [ERR_EMPTY]

    try:
        data = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return [ERR_JSON_PARSE]

    if not isinstance(data, dict):
        return [ERR_NOT_DICT]

    # Collect all error types from both shape attempts
    all_error_types: list[str] = []

    from model_2_training.src.schemas.workflow_schema import (
        MessageWorkflowResponse,
        ToolWorkflowResponse,
    )

    # Try Shape A
    try:
        ToolWorkflowResponse.model_validate(data)
        return []  # valid
    except Exception as e:
        all_error_types.extend(_extract_error_types(e))

    # Try Shape B
    try:
        MessageWorkflowResponse.model_validate(data)
        return []  # valid
    except Exception as e:
        all_error_types.extend(_extract_error_types(e))

    return list(set(all_error_types))


def _extract_error_types(exc: Exception) -> list[str]:
    """Extract Pydantic v2 error type strings from a ValidationError."""
    try:
        return [e["type"] for e in exc.errors()]  # type: ignore[attr-defined]
    except Exception:
        return [ERR_SCHEMA]


def _has_extra_fields(error_types: list[str]) -> bool:
    return any("extra_forbidden" in t for t in error_types)


def _has_wrong_type(error_types: list[str]) -> bool:
    type_errors = {
        "string_type", "dict_type", "list_type", "int_type",
        "bool_type", "float_type", "model_type",
    }
    return any(t in type_errors for t in error_types)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_schema_metrics(predictions: list[dict]) -> dict[str, float | int | dict]:
    """
    Compute Phase 2A Pydantic schema validation metrics over a list of predictions.

    Args:
        predictions: List of prediction dicts with at least "generated" and "reference" keys.

    Returns:
        Dict with:
          schema_valid_rate         — fraction passing full Pydantic validation
          correct_shape_rate        — fraction where generated shape matches reference shape
          extra_fields_rate         — fraction with unexpected extra keys at any level
          wrong_type_rate           — fraction where any field has the wrong Python type
          schema_error_distribution — {error_type: count} breakdown of failures
    """
    if not predictions:
        logger.warning("compute_schema_metrics called with empty predictions list.")
        return {}

    n = len(predictions)
    schema_valid = 0
    shape_correct = 0
    has_extra_fields = 0
    has_wrong_type = 0
    error_counter: Counter = Counter()

    for pred in predictions:
        generated = pred.get("generated", "") or ""
        reference = pred.get("reference", "") or ""

        # --- Schema validity ---
        error_types = _classify_pydantic_errors(generated)
        if not error_types:
            schema_valid += 1
        else:
            for et in error_types:
                error_counter[et] += 1
            if _has_extra_fields(error_types):
                has_extra_fields += 1
            if _has_wrong_type(error_types):
                has_wrong_type += 1

        # --- Shape correctness (vs reference, shape-only, no full validation needed) ---
        gen_shape = infer_shape(generated)
        ref_shape = infer_shape(reference)

        # Only count shape comparison when reference shape is determinable
        if ref_shape != OutputShape.UNKNOWN:
            if gen_shape == ref_shape:
                shape_correct += 1

    # Denominator for shape rate: only predictions where reference shape is known
    shape_denominator = sum(
        1 for p in predictions
        if infer_shape(p.get("reference", "") or "") != OutputShape.UNKNOWN
    )

    metrics: dict[str, float | int | dict] = {
        "schema_valid_rate": round(schema_valid / n, 4),
        "correct_shape_rate": round(shape_correct / shape_denominator, 4) if shape_denominator else 0.0,
        "extra_fields_rate": round(has_extra_fields / n, 4),
        "wrong_type_rate": round(has_wrong_type / n, 4),
        "schema_error_distribution": dict(error_counter.most_common()),
    }

    logger.info("Schema Metrics (Phase 2A):")
    for k, v in metrics.items():
        if k != "schema_error_distribution":
            logger.info(f"  {k:<30} {v}")
    if error_counter:
        logger.info("  Error distribution:")
        for err_type, count in error_counter.most_common():
            logger.info(f"    {err_type:<35} {count}")

    return metrics


def summarize_schema_errors(predictions: list[dict], n_samples: int = 10) -> list[dict]:
    """
    Return a sample of predictions that failed schema validation, for manual inspection.

    Args:
        predictions: Full predictions list.
        n_samples:   Max number of error examples to return.

    Returns:
        List of dicts with generated, reference, error_types, and json_parsed flag.
    """
    errors = []
    for pred in predictions:
        generated = pred.get("generated", "") or ""
        error_types = _classify_pydantic_errors(generated)
        if error_types:
            errors.append({
                "generated": generated[:500],
                "reference": (pred.get("reference", "") or "")[:300],
                "error_types": error_types,
                "json_parsed": ERR_JSON_PARSE not in error_types and ERR_EMPTY not in error_types,
            })
        if len(errors) >= n_samples:
            break
    return errors
