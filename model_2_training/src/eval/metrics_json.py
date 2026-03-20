"""
metrics_json.py

Computes JSON structural quality metrics on model-generated outputs.

These metrics measure whether the model is learning to produce consistent
structured (JSON-like) output — before full Pydantic schema validation is added.

Metrics implemented:
  1. non_empty_rate         — fraction of outputs that are non-empty strings
  2. starts_with_brace_rate — fraction starting with '{' or '['
  3. ends_with_brace_rate   — fraction ending with '}' or ']'
  4. json_parse_rate        — fraction that json.loads() parses successfully
  5. malformed_json_rate    — 1 - json_parse_rate
  6. avg_output_length      — mean character length of generated outputs
  7. truncation_rate        — fraction where output ends without closing brace
                              (proxy for hitting max_new_tokens limit)
  8. quote_balance_rate     — fraction where double-quote count is even
  9. brace_balance_rate     — fraction where '{' count equals '}' count
                              AND '[' count equals ']' count

All rates are floats in [0.0, 1.0].
"""

from __future__ import annotations
import json
from loguru import logger


def _is_non_empty(text: str) -> bool:
    return isinstance(text, str) and len(text.strip()) > 0


def _starts_with_brace(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("{") or stripped.startswith("[")


def _ends_with_brace(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith("}") or stripped.endswith("]")


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _is_truncated(text: str) -> bool:
    """Heuristic: output looks truncated if it starts with { or [ but doesn't close."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("{") and not stripped.endswith("}"):
        return True
    if stripped.startswith("[") and not stripped.endswith("]"):
        return True
    return False


def _has_balanced_quotes(text: str) -> bool:
    return text.count('"') % 2 == 0


def _has_balanced_braces(text: str) -> bool:
    return text.count("{") == text.count("}") and text.count("[") == text.count("]")


def compute_metrics(predictions: list[dict]) -> dict[str, float | int]:
    """
    Compute all JSON structural quality metrics over a list of predictions.

    Args:
        predictions: List of prediction dicts with at least a "generated" key.

    Returns:
        Dict of metric_name -> value.
    """
    if not predictions:
        logger.warning("compute_metrics called with empty predictions list.")
        return {}

    generated_texts = [p.get("generated", "") or "" for p in predictions]
    n = len(generated_texts)

    non_empty = sum(_is_non_empty(t) for t in generated_texts)
    starts_brace = sum(_starts_with_brace(t) for t in generated_texts if _is_non_empty(t))
    ends_brace = sum(_ends_with_brace(t) for t in generated_texts if _is_non_empty(t))
    json_parse = sum(_is_valid_json(t) for t in generated_texts)
    truncated = sum(_is_truncated(t) for t in generated_texts)
    balanced_quotes = sum(_has_balanced_quotes(t) for t in generated_texts if _is_non_empty(t))
    balanced_braces = sum(_has_balanced_braces(t) for t in generated_texts if _is_non_empty(t))

    non_empty_n = max(non_empty, 1)  # avoid division by zero in sub-rates
    lengths = [len(t) for t in generated_texts if _is_non_empty(t)]
    avg_length = sum(lengths) / len(lengths) if lengths else 0.0

    metrics = {
        "total_samples": n,
        "non_empty_count": non_empty,
        "non_empty_rate": round(non_empty / n, 4),
        "starts_with_brace_rate": round(starts_brace / non_empty_n, 4),
        "ends_with_brace_rate": round(ends_brace / non_empty_n, 4),
        "json_parse_rate": round(json_parse / n, 4),
        "malformed_json_rate": round(1.0 - json_parse / n, 4),
        "avg_output_length": round(avg_length, 2),
        "truncation_rate": round(truncated / n, 4),
        "quote_balance_rate": round(balanced_quotes / non_empty_n, 4),
        "brace_balance_rate": round(balanced_braces / non_empty_n, 4),
    }

    logger.info("JSON Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k:<30} {v}")

    return metrics


def summarize_errors(predictions: list[dict], n_samples: int = 10) -> list[dict]:
    """
    Return a sample of predictions where JSON parsing failed, for manual inspection.

    Args:
        predictions: Full predictions list.
        n_samples: Max number of error examples to return.

    Returns:
        List of dicts with "generated", "reference", "error_reason".
    """
    errors = []
    for p in predictions:
        text = p.get("generated", "") or ""
        if not _is_valid_json(text):
            errors.append({
                "generated": text[:500],  # truncate for readability
                "reference": (p.get("reference") or "")[:300],
                "error_reason": "json_parse_failed",
                "is_empty": not _is_non_empty(text),
                "starts_brace": _starts_with_brace(text),
                "ends_brace": _ends_with_brace(text),
            })
        if len(errors) >= n_samples:
            break
    return errors
