"""
slice_eval.py

Phase 3 — Slice-based evaluation.

Groups predictions into meaningful subsets and computes metrics per subset.
High overall accuracy can hide poor performance on specific archetypes,
datasets, or input sizes. Slice evaluation reveals exactly where the model
struggles.

Slice dimensions
----------------
  archetype      multi_step / single_step / refusal  (derived from reference)
  dataset        From sample.source_dataset / metadata.source (or "unknown")
  input_length   short (<100 chars) / medium (100–300) / long (>300)
  complexity     refusal / simple (1 step) / moderate (2–3 steps) / complex (4+)
  error_category VALID / EMPTY / TRUNCATED / MISSING_WORKFLOW / … (error taxonomy)

Key metrics tracked per slice (KEY_METRICS)
-------------------------------------------
  phase1_structural/json_parse_rate
  phase1_structural/tax_valid_rate
  phase2a_schema/schema_valid_rate
  phase2b_fields/steps_count_match_rate
  phase2c_params/full_param_validity_rate

Public API
----------
  compute_slice_metrics(predictions, slice_by) -> dict[slice_value, metrics]
  compute_all_slices(predictions)              -> dict[dimension, dict[slice_value, metrics]]
  compute_robustness_delta(clean, perturbed)   -> dict[metric, delta]
  build_robustness_summary(clean, perturbed_by_type) -> dict
"""

from __future__ import annotations

import json
from collections import defaultdict
from loguru import logger

from model_2_training.src.eval.metrics_aggregator import run_all_metrics
from model_2_training.src.eval.error_taxonomy import _classify as _classify_output


# ---------------------------------------------------------------------------
# Key metrics tracked in robustness comparisons
# ---------------------------------------------------------------------------

KEY_METRICS: list[str] = [
    "phase1_structural/json_parse_rate",
    "phase1_structural/tax_valid_rate",
    "phase2a_schema/schema_valid_rate",
    "phase2b_fields/steps_count_match_rate",
    "phase2c_params/full_param_validity_rate",
]


# ---------------------------------------------------------------------------
# Slice extraction helpers
# ---------------------------------------------------------------------------

def _get_archetype(pred: dict) -> str:
    """Classify a prediction by the archetype of its reference answer."""
    reference = pred.get("reference", "") or ""
    try:
        parsed = json.loads(reference.strip())
        steps = parsed.get("workflow", {}).get("steps", None)
        if steps is None:
            return "no_workflow"
        if not isinstance(steps, list):
            return "invalid_steps"
        if len(steps) == 0:
            return "refusal"
        if len(steps) == 1:
            return "single_step"
        return "multi_step"
    except (json.JSONDecodeError, ValueError):
        return "invalid_json"


def _get_dataset(pred: dict) -> str:
    """
    Extract the dataset source label from sample metadata.

    Checks metadata.source first, then metadata.dataset, then the top-level
    source_dataset key written by combine.py, then falls back to "unknown".
    """
    sample = pred.get("sample") or {}
    metadata = sample.get("metadata") or {}
    source = (
        metadata.get("source")
        or metadata.get("dataset")
        or sample.get("source_dataset")
        or "unknown"
    )
    return str(source)


def _get_input_length_bucket(pred: dict) -> str:
    """
    Bucket predictions by the character length of the user message.

      short  : < 100 chars
      medium : 100–300 chars
      long   : > 300 chars
    """
    sample = pred.get("sample") or {}
    messages = sample.get("messages") or []
    user_content = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break
    length = len(user_content)
    if length < 100:
        return "short"
    elif length <= 300:
        return "medium"
    else:
        return "long"


def _get_complexity(pred: dict) -> str:
    """
    Classify task complexity from the reference step count.

      refusal  : workflow.steps = []
      simple   : 1 step
      moderate : 2–3 steps
      complex  : 4+ steps
    """
    reference = pred.get("reference", "") or ""
    try:
        parsed = json.loads(reference.strip())
        steps = parsed.get("workflow", {}).get("steps", None)
        if not isinstance(steps, list):
            return "unknown"
        n = len(steps)
        if n == 0:
            return "refusal"
        if n == 1:
            return "simple"
        if n <= 3:
            return "moderate"
        return "complex"
    except (json.JSONDecodeError, ValueError):
        return "unknown"


def _get_error_category(pred: dict) -> str:
    """Classify prediction by its failure category from the error taxonomy."""
    generated = pred.get("generated", "") or ""
    reference = pred.get("reference", "") or ""
    return _classify_output(generated, reference)


# _get_error_category must be defined before this dict
_SLICE_EXTRACTORS: dict = {
    "archetype":      _get_archetype,
    "dataset":        _get_dataset,
    "input_length":   _get_input_length_bucket,
    "complexity":     _get_complexity,
    "error_category": _get_error_category,
}


# ---------------------------------------------------------------------------
# Core slice evaluation
# ---------------------------------------------------------------------------

def compute_slice_metrics(
    predictions: list[dict],
    slice_by: str,
) -> dict[str, dict]:
    """
    Group predictions by a slice dimension and compute metrics per group.

    Each group is passed through the full run_all_metrics() pipeline,
    so every group produces the same metric keys as a full eval run.
    A "slice_n" key is added to each group's metrics dict for convenience.

    Args:
        predictions: List of prediction dicts from run_generation().
        slice_by:    Slice dimension — one of the keys in _SLICE_EXTRACTORS.

    Returns:
        Dict mapping slice_value -> metrics_dict.

    Raises:
        ValueError: If slice_by is not a known dimension.
    """
    if slice_by not in _SLICE_EXTRACTORS:
        raise ValueError(
            f"Unknown slice dimension: '{slice_by}'. "
            f"Valid dimensions: {list(_SLICE_EXTRACTORS.keys())}"
        )

    extractor = _SLICE_EXTRACTORS[slice_by]
    groups: dict[str, list[dict]] = defaultdict(list)
    for pred in predictions:
        groups[extractor(pred)].append(pred)

    counts = {k: len(v) for k, v in groups.items()}
    logger.info(f"Slice '{slice_by}': {len(groups)} groups — {counts}")

    slice_metrics: dict[str, dict] = {}
    for slice_val, group in sorted(groups.items()):
        logger.info(f"  slice {slice_by}={slice_val!r}  n={len(group)}")
        try:
            metrics = run_all_metrics(group)
            metrics["slice_n"] = len(group)
            slice_metrics[slice_val] = metrics
        except Exception as e:
            logger.warning(f"  Metrics failed for {slice_by}={slice_val!r}: {e}")
            slice_metrics[slice_val] = {"slice_n": len(group), "error": str(e)}

    return slice_metrics


def compute_all_slices(predictions: list[dict]) -> dict[str, dict[str, dict]]:
    """
    Run slice evaluation across all five slice dimensions.

    Args:
        predictions: List of prediction dicts from run_generation().

    Returns:
        Dict mapping dimension_name -> {slice_value -> metrics_dict}.
    """
    results: dict[str, dict[str, dict]] = {}
    for dim in _SLICE_EXTRACTORS:
        logger.info(f"Computing slices for dimension: {dim}")
        results[dim] = compute_slice_metrics(predictions, dim)
    return results


# ---------------------------------------------------------------------------
# Robustness delta helpers
# ---------------------------------------------------------------------------

def compute_robustness_delta(
    clean_metrics: dict,
    perturbed_metrics: dict,
) -> dict[str, float]:
    """
    Compute performance degradation between clean and perturbed evaluation.

    Delta = clean_value - perturbed_value for each key metric.
    Positive delta = degradation (clean was better).
    Negative delta = improvement (unusual; likely noise on small samples).

    Args:
        clean_metrics:     Metrics dict from run_all_metrics() on clean data.
        perturbed_metrics: Metrics dict from run_all_metrics() on perturbed data.

    Returns:
        Dict mapping metric_key -> delta (float). Only includes metrics that
        exist in both dicts.
    """
    delta: dict[str, float] = {}
    for key in KEY_METRICS:
        c = clean_metrics.get(key)
        p = perturbed_metrics.get(key)
        if c is not None and p is not None:
            delta[key] = round(float(c) - float(p), 4)
    return delta


def build_robustness_summary(
    clean_metrics: dict,
    perturbed_by_type: dict[str, dict],
) -> dict:
    """
    Build a summary table comparing clean vs every perturbation type.

    Args:
        clean_metrics:      Metrics on unperturbed samples.
        perturbed_by_type:  Dict mapping perturbation_type -> metrics_dict.

    Returns:
        Dict with:
          "clean"                          : {metric: clean_value}
          "perturbed"                      : {pert_type: {metric: value}}
          "delta"                          : {pert_type: {metric: delta}}
          "worst_perturbation"             : perturbation type with highest total delta
          "worst_perturbation_total_delta" : sum of all key metric deltas for the worst type
    """
    summary: dict = {
        "clean":     {k: clean_metrics.get(k) for k in KEY_METRICS},
        "perturbed": {},
        "delta":     {},
    }

    max_total_delta = float("-inf")
    worst_perturbation: str | None = None

    for pert_type, pert_metrics in perturbed_by_type.items():
        summary["perturbed"][pert_type] = {k: pert_metrics.get(k) for k in KEY_METRICS}
        delta = compute_robustness_delta(clean_metrics, pert_metrics)
        summary["delta"][pert_type] = delta

        total = sum(v for v in delta.values())
        if total > max_total_delta:
            max_total_delta = total
            worst_perturbation = pert_type

    summary["worst_perturbation"] = worst_perturbation
    summary["worst_perturbation_total_delta"] = round(max_total_delta, 4)
    return summary