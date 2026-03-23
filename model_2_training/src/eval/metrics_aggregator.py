"""
metrics_aggregator.py

Phase 2D — Unified metrics pipeline.

Single entry point that runs all evaluation phases in order and returns one
merged metrics dict consumed by save_reports.py, wandb_logger.py, and the
run_eval / run_test scripts.

Phase execution order:
  Phase 1  — JSON structural quality  (metrics_json)
  Phase 2A — Pydantic schema validity (metrics_schema)
  Phase 2B — Field-level correctness  (metrics_fields)
  Phase 2C — Parameter validation     (metrics_params)

All phases receive the full predictions list. Phases are independent modules —
a failure in one does not stop the others. Each phase's metrics are namespaced
with a phase prefix in the returned dict for clarity in W&B and reports.

Returns a flat dict of all metrics (no nesting) so it can be logged directly
to W&B via wandb.log() and written to metrics.json without further processing.
"""

from __future__ import annotations

from loguru import logger

from model_2_training.src.eval.metrics_json import compute_metrics as _phase1_metrics
from model_2_training.src.eval.metrics_schema import (
    compute_schema_metrics as _phase2a_metrics,
    summarize_schema_errors,
)
from model_2_training.src.eval.metrics_fields import (
    compute_field_metrics as _phase2b_metrics,
    compute_per_field_report,
)
from model_2_training.src.eval.metrics_params import (
    compute_param_metrics as _phase2c_metrics,
    summarize_param_errors,
)


# ---------------------------------------------------------------------------
# Phase prefixes — used in W&B and reports
# ---------------------------------------------------------------------------

_PHASE_LABELS = {
    "p1":  "phase1_structural",
    "p2a": "phase2a_schema",
    "p2b": "phase2b_fields",
    "p2c": "phase2c_params",
}


def run_all_metrics(predictions: list[dict]) -> dict:
    """
    Run all evaluation phases and return a single merged metrics dict.

    Args:
        predictions: List of prediction dicts from generator.run_generation().
                     Each dict must have at minimum: "generated", "reference".
                     Phase 2C also uses the "sample" key for tool schema extraction.

    Returns:
        Flat dict with all metrics from all phases. Keys are prefixed by phase:
          phase1_structural/*   — JSON structural metrics
          phase2a_schema/*      — Pydantic schema validation metrics
          phase2b_fields/*      — field-level correctness metrics
          phase2c_params/*      — parameter validation metrics

        Also includes top-level "total_samples" for convenience.
    """
    if not predictions:
        logger.warning("run_all_metrics called with empty predictions list.")
        return {}

    merged: dict = {"total_samples": len(predictions)}

    # --- Phase 1 ---
    try:
        p1 = _phase1_metrics(predictions)
        merged.update({f"phase1_structural/{k}": v for k, v in p1.items()})
        logger.info(f"Phase 1 complete — {len(p1)} metrics")
    except Exception as e:
        logger.error(f"Phase 1 metrics failed: {e}")

    # --- Phase 2A ---
    try:
        p2a = _phase2a_metrics(predictions)
        # schema_error_distribution is a dict — keep it nested under its own key
        for k, v in p2a.items():
            if k == "schema_error_distribution":
                merged["phase2a_schema/schema_error_distribution"] = v
            else:
                merged[f"phase2a_schema/{k}"] = v
        logger.info(f"Phase 2A complete — {len(p2a)} metrics")
    except Exception as e:
        logger.error(f"Phase 2A metrics failed: {e}")

    # --- Phase 2B ---
    try:
        p2b = _phase2b_metrics(predictions)
        merged.update({f"phase2b_fields/{k}": v for k, v in p2b.items()})
        logger.info(f"Phase 2B complete — {len(p2b)} metrics")
    except Exception as e:
        logger.error(f"Phase 2B metrics failed: {e}")

    # --- Phase 2C ---
    try:
        p2c = _phase2c_metrics(predictions)
        merged.update({f"phase2c_params/{k}": v for k, v in p2c.items()})
        logger.info(f"Phase 2C complete — {len(p2c)} metrics")
    except Exception as e:
        logger.error(f"Phase 2C metrics failed: {e}")

    logger.success(f"All phases complete — {len(merged)} total metrics")
    return merged


def get_per_field_report(predictions: list[dict]) -> dict[str, dict]:
    """
    Return the per-field breakdown report (Phase 2B).
    Separated from run_all_metrics so it can be passed to save_reports.py.
    """
    try:
        return compute_per_field_report(predictions)
    except Exception as e:
        logger.warning(f"Per-field report failed: {e}")
        return {}


def get_schema_errors(predictions: list[dict], n_samples: int = 10) -> list[dict]:
    """Return Phase 2A schema error samples for the error report."""
    try:
        return summarize_schema_errors(predictions, n_samples=n_samples)
    except Exception as e:
        logger.warning(f"Schema error summary failed: {e}")
        return []


def get_param_errors(predictions: list[dict], n_samples: int = 10) -> list[dict]:
    """Return Phase 2C param error samples for the error report."""
    try:
        return summarize_param_errors(predictions, n_samples=n_samples)
    except Exception as e:
        logger.warning(f"Param error summary failed: {e}")
        return []
