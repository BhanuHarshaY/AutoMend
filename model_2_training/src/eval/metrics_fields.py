"""
metrics_fields.py

Phase 2B — Field-level correctness metrics.

Operates on raw JSON-parsed dicts (not Pydantic models) so it catches value
issues even in outputs that failed Phase 2A schema validation. Each check
is only counted over the subset of predictions where that field is applicable.

Metrics:
  tool_name_nonempty_rate   — fraction of steps where tool name is a non-empty string
  params_nonempty_rate      — fraction of steps where params dict is non-empty
  steps_count_match_rate    — fraction of predictions where step count == reference step count
  message_nonempty_rate     — fraction of message-type responses where message is non-empty
  steps_is_list_rate        — fraction of parseable predictions where steps is actually a list

Also provides compute_per_field_report() for a per-field breakdown table.

All rates are floats in [0.0, 1.0].
"""

from __future__ import annotations

import json
from loguru import logger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_parse(text: str) -> dict | None:
    """Parse text as JSON dict. Returns None on any failure."""
    if not text or not text.strip():
        return None
    try:
        data = json.loads(text.strip())
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _get_steps(data: dict) -> list | None:
    """Extract steps list from a parsed dict. Returns None if path is missing or wrong type."""
    workflow = data.get("workflow")
    if not isinstance(workflow, dict):
        return None
    steps = workflow.get("steps")
    return steps if isinstance(steps, list) else None


def _is_tool_shape(data: dict) -> bool:
    """True if parsed dict looks like a tool workflow (non-empty steps)."""
    steps = _get_steps(data)
    return bool(steps)


def _is_message_shape(data: dict) -> bool:
    """True if parsed dict looks like a message response (empty steps + message key)."""
    steps = _get_steps(data)
    return isinstance(steps, list) and len(steps) == 0 and "message" in data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_field_metrics(predictions: list[dict]) -> dict[str, float | int]:
    """
    Compute Phase 2B field-level correctness metrics over a list of predictions.

    Args:
        predictions: List of prediction dicts with "generated" and "reference" keys.

    Returns:
        Dict with rates (floats in [0.0, 1.0]) and denominator counts for transparency.
    """
    if not predictions:
        logger.warning("compute_field_metrics called with empty predictions list.")
        return {}

    # Counters (numerator, denominator) per check
    tool_nonempty_pass = 0
    tool_nonempty_total = 0

    params_nonempty_pass = 0
    params_nonempty_total = 0

    steps_count_pass = 0
    steps_count_total = 0

    message_nonempty_pass = 0
    message_nonempty_total = 0

    steps_is_list_pass = 0
    steps_is_list_total = 0

    for pred in predictions:
        generated = pred.get("generated", "") or ""
        reference = pred.get("reference", "") or ""

        gen_data = _try_parse(generated)
        ref_data = _try_parse(reference)

        # ---- steps_is_list_rate (all parseable predictions) ----
        if gen_data is not None:
            steps_is_list_total += 1
            workflow = gen_data.get("workflow")
            if isinstance(workflow, dict) and isinstance(workflow.get("steps"), list):
                steps_is_list_pass += 1

        # ---- step-level checks (only for tool-shape predictions) ----
        if gen_data is not None:
            steps = _get_steps(gen_data)
            if steps:  # non-empty list → tool shape
                for step in steps:
                    if not isinstance(step, dict):
                        # malformed step — count as failures for both checks
                        tool_nonempty_total += 1
                        params_nonempty_total += 1
                        continue

                    # tool_name_nonempty_rate
                    tool_nonempty_total += 1
                    tool_val = step.get("tool")
                    if isinstance(tool_val, str) and tool_val.strip():
                        tool_nonempty_pass += 1

                    # params_nonempty_rate
                    params_nonempty_total += 1
                    params_val = step.get("params")
                    if isinstance(params_val, dict) and len(params_val) > 0:
                        params_nonempty_pass += 1

        # ---- steps_count_match_rate (both sides must parse) ----
        if gen_data is not None and ref_data is not None:
            gen_steps = _get_steps(gen_data)
            ref_steps = _get_steps(ref_data)
            if gen_steps is not None and ref_steps is not None:
                steps_count_total += 1
                if len(gen_steps) == len(ref_steps):
                    steps_count_pass += 1

        # ---- message_nonempty_rate (message-shape predictions only) ----
        if gen_data is not None and _is_message_shape(gen_data):
            message_nonempty_total += 1
            msg = gen_data.get("message")
            if isinstance(msg, str) and msg.strip():
                message_nonempty_pass += 1

    def _rate(num: int, denom: int) -> float:
        return round(num / denom, 4) if denom > 0 else 0.0

    metrics: dict[str, float | int] = {
        "tool_name_nonempty_rate": _rate(tool_nonempty_pass, tool_nonempty_total),
        "tool_name_nonempty_denominator": tool_nonempty_total,
        "params_nonempty_rate": _rate(params_nonempty_pass, params_nonempty_total),
        "params_nonempty_denominator": params_nonempty_total,
        "steps_count_match_rate": _rate(steps_count_pass, steps_count_total),
        "steps_count_match_denominator": steps_count_total,
        "message_nonempty_rate": _rate(message_nonempty_pass, message_nonempty_total),
        "message_nonempty_denominator": message_nonempty_total,
        "steps_is_list_rate": _rate(steps_is_list_pass, steps_is_list_total),
        "steps_is_list_denominator": steps_is_list_total,
    }

    logger.info("Field Metrics (Phase 2B):")
    rate_keys = [k for k in metrics if k.endswith("_rate")]
    for k in rate_keys:
        denom_key = k.replace("_rate", "_denominator")
        denom = metrics.get(denom_key, "?")
        logger.info(f"  {k:<35} {metrics[k]}  (n={denom})")

    return metrics


def compute_per_field_report(predictions: list[dict]) -> dict[str, dict]:
    """
    Return a per-field breakdown showing present/non_empty/correct_type rates.

    Only covers fields that appear in the output schema: tool, params, message, steps.
    Each sub-metric is computed over predictions where the field is applicable.

    Returns:
        {
          "tool":    {"present": float, "non_empty": float, "correct_type": float},
          "params":  {"present": float, "non_empty": float, "correct_type": float},
          "message": {"present": float, "non_empty": float, "correct_type": float},
          "steps":   {"present": float, "is_list": float},
        }
    """
    if not predictions:
        return {}

    report: dict[str, dict] = {
        "tool":    {"present_n": 0, "present_pass": 0, "non_empty_pass": 0, "correct_type_pass": 0},
        "params":  {"present_n": 0, "present_pass": 0, "non_empty_pass": 0, "correct_type_pass": 0},
        "message": {"present_n": 0, "present_pass": 0, "non_empty_pass": 0, "correct_type_pass": 0},
        "steps":   {"present_n": 0, "present_pass": 0, "is_list_pass": 0},
    }

    for pred in predictions:
        gen_data = _try_parse(pred.get("generated", "") or "")
        if gen_data is None:
            continue

        workflow = gen_data.get("workflow")

        # ---- steps ----
        report["steps"]["present_n"] += 1
        if isinstance(workflow, dict):
            report["steps"]["present_pass"] += 1
            steps = workflow.get("steps")
            if isinstance(steps, list):
                report["steps"]["is_list_pass"] += 1

            # ---- step-level fields ----
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        continue

                    # tool
                    report["tool"]["present_n"] += 1
                    tool_val = step.get("tool")
                    if tool_val is not None:
                        report["tool"]["present_pass"] += 1
                    if isinstance(tool_val, str):
                        report["tool"]["correct_type_pass"] += 1
                        if tool_val.strip():
                            report["tool"]["non_empty_pass"] += 1

                    # params
                    report["params"]["present_n"] += 1
                    params_val = step.get("params")
                    if params_val is not None:
                        report["params"]["present_pass"] += 1
                    if isinstance(params_val, dict):
                        report["params"]["correct_type_pass"] += 1
                        if params_val:
                            report["params"]["non_empty_pass"] += 1

        # ---- message (message-shape only) ----
        if _is_message_shape(gen_data):
            report["message"]["present_n"] += 1
            msg = gen_data.get("message")
            if msg is not None:
                report["message"]["present_pass"] += 1
            if isinstance(msg, str):
                report["message"]["correct_type_pass"] += 1
                if msg.strip():
                    report["message"]["non_empty_pass"] += 1

    def _r(num: int, denom: int) -> float:
        return round(num / denom, 4) if denom > 0 else 0.0

    return {
        "tool": {
            "present":      _r(report["tool"]["present_pass"],      report["tool"]["present_n"]),
            "non_empty":    _r(report["tool"]["non_empty_pass"],     report["tool"]["present_n"]),
            "correct_type": _r(report["tool"]["correct_type_pass"],  report["tool"]["present_n"]),
        },
        "params": {
            "present":      _r(report["params"]["present_pass"],     report["params"]["present_n"]),
            "non_empty":    _r(report["params"]["non_empty_pass"],    report["params"]["present_n"]),
            "correct_type": _r(report["params"]["correct_type_pass"], report["params"]["present_n"]),
        },
        "message": {
            "present":      _r(report["message"]["present_pass"],     report["message"]["present_n"]),
            "non_empty":    _r(report["message"]["non_empty_pass"],   report["message"]["present_n"]),
            "correct_type": _r(report["message"]["correct_type_pass"],report["message"]["present_n"]),
        },
        "steps": {
            "present": _r(report["steps"]["present_pass"], report["steps"]["present_n"]),
            "is_list": _r(report["steps"]["is_list_pass"], report["steps"]["present_n"]),
        },
    }
