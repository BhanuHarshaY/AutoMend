"""
fairlearn_check.py

Model bias detection using Fairlearn and data slicing.

Ensures the model does not ignore failures on background systems while
only catching failures on user-facing systems. Slices predictions by
`source_dataset` and optionally by derived compute_tier / namespace features.

Uses fairlearn.metrics.MetricFrame to compute per-slice metrics and
detect disparities. If bias is detected, recommends WeightedRandomSampler
configuration for mitigation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, recall_score

from fairlearn.metrics import MetricFrame


LABEL_NAMES = [
    "Normal", "Resource_Exhaustion", "System_Crash",
    "Network_Failure", "Data_Drift", "Auth_Failure", "Permission_Denied",
]


def _macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _macro_recall(y_true, y_pred) -> float:
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


def _anomaly_recall(y_true, y_pred) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return 0.0
    return float(recall_score(y_true[mask], y_pred[mask], average="macro", zero_division=0))


DEFAULT_METRICS: Dict[str, Callable] = {
    "macro_f1": _macro_f1,
    "macro_recall": _macro_recall,
    "anomaly_recall": _anomaly_recall,
}


def run_bias_detection(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.DataFrame,
    metrics: Optional[Dict[str, Callable]] = None,
    disparity_threshold: float = 0.10,
) -> Dict:
    """
    Run Fairlearn bias detection across data slices.

    Args:
        y_true: Ground truth labels.
        y_pred: Model predictions.
        sensitive_features: DataFrame with slice columns (e.g., source_dataset).
        metrics: Dict of metric_name -> callable(y_true, y_pred).
        disparity_threshold: Max acceptable disparity between groups.

    Returns:
        Dict with per-slice metrics, disparities, and bias flags.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    results: Dict = {"slices": {}, "disparity": {}, "bias_detected": False, "details": []}

    for col in sensitive_features.columns:
        logger.info(f"Analyzing bias across '{col}' slices...")
        groups = sensitive_features[col]

        mf = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=groups,
        )

        by_group = mf.by_group.to_dict()

        col_slices = {}
        for metric_name in metrics:
            group_vals = by_group.get(metric_name, {})
            for group_name, value in group_vals.items():
                if group_name not in col_slices:
                    col_slices[group_name] = {}
                col_slices[group_name][metric_name] = float(value) if value is not None else 0.0

        results["slices"][col] = col_slices

        diff = mf.difference()
        for metric_name, disp_val in diff.items():
            disp = float(disp_val)
            results["disparity"][f"{col}/{metric_name}"] = disp

            if disp > disparity_threshold:
                results["bias_detected"] = True
                results["details"].append({
                    "column": col,
                    "metric": metric_name,
                    "disparity": disp,
                    "threshold": disparity_threshold,
                    "message": f"Disparity of {disp:.4f} exceeds threshold {disparity_threshold}",
                })
                logger.warning(
                    f"BIAS DETECTED: {col}/{metric_name} disparity = {disp:.4f} "
                    f"(threshold: {disparity_threshold})"
                )

        logger.info(f"  Slice metrics for '{col}':")
        for group_name, group_metrics in col_slices.items():
            logger.info(f"    {group_name}: {group_metrics}")

    return results


def compute_sampler_weights(
    sources: np.ndarray | List[str],
    y_true: np.ndarray,
) -> np.ndarray:
    """
    Compute per-sample weights for WeightedRandomSampler to upsample
    underrepresented source datasets and anomaly classes.

    Args:
        sources: Array of source_dataset values per sample.
        y_true: Ground truth labels.

    Returns:
        Per-sample weight array for torch.utils.data.WeightedRandomSampler.
    """
    sources = np.array(sources)
    y_true = np.array(y_true)

    unique_sources, source_counts = np.unique(sources, return_counts=True)
    source_weight = {s: len(sources) / c for s, c in zip(unique_sources, source_counts)}

    unique_labels, label_counts = np.unique(y_true, return_counts=True)
    label_weight = {l: len(y_true) / c for l, c in zip(unique_labels, label_counts)}

    weights = np.array([
        source_weight[s] * label_weight[l]
        for s, l in zip(sources, y_true)
    ])

    weights = weights / weights.sum() * len(weights)

    logger.info(f"Sampler weights — source weights: {source_weight}")
    logger.info(f"Sampler weights — label weights: {label_weight}")
    return weights


def save_bias_report(results: Dict, output_dir: str | Path) -> Path:
    """Save the bias detection report to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "bias_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Bias report -> {out_path}")
    return out_path
