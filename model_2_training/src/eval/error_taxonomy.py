"""
error_taxonomy.py

Labels each model prediction with a failure category from a fixed taxonomy.

Taxonomy (mutually exclusive, applied in priority order):
  VALID               — output is non-empty and parses as valid JSON
  EMPTY               — output is empty or whitespace only
  TRUNCATED           — starts with { or [ but missing closing brace/bracket
  MISSING_WORKFLOW    — valid JSON but no top-level "workflow" key
  WRONG_STEPS_TYPE    — "workflow" present but "steps" is not a list
  EMPTY_STEPS         — valid JSON with workflow.steps = [] (refusal or no-op)
  MALFORMED_JSON      — non-empty, not truncated, but json.loads() fails
  UNBALANCED_BRACES   — { / } or [ / ] counts don't match
  UNBALANCED_QUOTES   — double-quote count is odd (broken string literal)

Each prediction dict gets a "failure_category" key added in place.
"""

from __future__ import annotations
import json
from loguru import logger

# Ordered labels — first match wins
TAXONOMY_LABELS = [
    "VALID",
    "EMPTY",
    "TRUNCATED",
    "MISSING_WORKFLOW",
    "WRONG_STEPS_TYPE",
    "EMPTY_STEPS",
    "MALFORMED_JSON",
    "UNBALANCED_BRACES",
    "UNBALANCED_QUOTES",
]


def _reference_is_refusal(reference: str) -> bool:
    """Return True if the reference answer itself has workflow.steps = []."""
    try:
        parsed = json.loads(reference.strip())
        steps = parsed.get("workflow", {}).get("steps", None)
        return isinstance(steps, list) and len(steps) == 0
    except (json.JSONDecodeError, ValueError, AttributeError):
        return False


def _classify(text: str, reference: str = "") -> str:
    """Return the single failure-category label for one generated output."""
    if not isinstance(text, str) or not text.strip():
        return "EMPTY"

    stripped = text.strip()

    # Try parsing first — if it works, check schema
    try:
        parsed = json.loads(stripped)
        # Valid JSON — now check structural schema
        if not isinstance(parsed, dict) or "workflow" not in parsed:
            return "MISSING_WORKFLOW"
        workflow = parsed["workflow"]
        if not isinstance(workflow, dict) or "steps" not in workflow:
            return "MISSING_WORKFLOW"
        steps = workflow["steps"]
        if not isinstance(steps, list):
            return "WRONG_STEPS_TYPE"
        if len(steps) == 0:
            # Only penalize empty steps if the reference expected non-empty steps
            if _reference_is_refusal(reference):
                return "VALID"   # model correctly refused
            return "EMPTY_STEPS"
        return "VALID"
    except (json.JSONDecodeError, ValueError):
        pass

    # Not valid JSON — classify the structural failure
    opens_with_brace = stripped.startswith("{") or stripped.startswith("[")
    closes_with_brace = stripped.endswith("}") or stripped.endswith("]")

    if opens_with_brace and not closes_with_brace:
        return "TRUNCATED"

    if stripped.count("{") != stripped.count("}") or stripped.count("[") != stripped.count("]"):
        return "UNBALANCED_BRACES"

    if stripped.count('"') % 2 != 0:
        return "UNBALANCED_QUOTES"

    return "MALFORMED_JSON"


def label_predictions(predictions: list[dict]) -> list[dict]:
    """
    Add a "failure_category" field to each prediction dict in-place.

    Args:
        predictions: List of prediction dicts with at least a "generated" key.

    Returns:
        The same list with "failure_category" added to each entry.
    """
    for pred in predictions:
        text      = pred.get("generated", "") or ""
        reference = pred.get("reference", "") or ""
        pred["failure_category"] = _classify(text, reference)
    return predictions


def compute_taxonomy_counts(predictions: list[dict]) -> dict[str, int | float]:
    """
    Count predictions per failure category and compute rates.

    Expects predictions to already have "failure_category" set
    (call label_predictions first).

    Returns:
        Dict with raw counts and rates for every taxonomy label,
        plus "taxonomy_total".
    """
    if not predictions:
        return {}

    n = len(predictions)
    counts: dict[str, int] = {label: 0 for label in TAXONOMY_LABELS}

    for pred in predictions:
        cat = pred.get("failure_category", "MALFORMED_JSON")
        if cat in counts:
            counts[cat] += 1
        else:
            counts["MALFORMED_JSON"] += 1

    result: dict[str, int | float] = {"taxonomy_total": n}
    for label in TAXONOMY_LABELS:
        result[f"tax_{label.lower()}_count"] = counts[label]
        result[f"tax_{label.lower()}_rate"] = round(counts[label] / n, 4)

    logger.info("Error Taxonomy Breakdown:")
    for label in TAXONOMY_LABELS:
        cnt = counts[label]
        rate = cnt / n
        logger.info(f"  {label:<22} {cnt:>4}  ({rate * 100:.1f}%)")

    return result


def get_taxonomy_errors(predictions: list[dict], exclude: tuple[str, ...] = ("VALID",)) -> list[dict]:
    """
    Return predictions whose failure_category is not in `exclude`.

    Useful for saving only the actual failure cases to error_samples.json.

    Args:
        predictions: Labeled predictions (label_predictions must have run).
        exclude:     Categories to exclude (default: exclude VALID outputs).

    Returns:
        Filtered list of prediction dicts with failure details.
    """
    errors = []
    for pred in predictions:
        cat = pred.get("failure_category")
        if cat not in exclude:
            errors.append({
                "index":            pred.get("index"),
                "failure_category": cat,
                "generated":        (pred.get("generated") or "")[:500],
                "reference":        (pred.get("reference") or "")[:300],
            })
    return errors
