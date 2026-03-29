"""
plots.py

Visualization utilities for Model 1 evaluation results.
Generates confusion matrix heatmaps and per-class metric bar charts.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

LABEL_NAMES: List[str] = [
    "Normal",
    "Resource_Exhaustion",
    "System_Crash",
    "Network_Failure",
    "Data_Drift",
    "Auth_Failure",
    "Permission_Denied",
]


def plot_confusion_matrix(
    cm: np.ndarray | list,
    output_path: str | Path,
    label_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    dpi: int = 150,
) -> Path:
    """
    Save a confusion matrix heatmap to disk.

    Args:
        cm: Confusion matrix (num_labels x num_labels).
        output_path: File path for the saved image.
        label_names: Class names for axis labels.
        title: Plot title.
        dpi: Image resolution.

    Returns:
        Path to the saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.array(cm)
    names = label_names or LABEL_NAMES[:cm.shape[0]]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=names, yticklabels=names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Confusion matrix saved -> {output_path}")
    return output_path


def plot_per_class_metrics(
    per_class: Dict[str, Dict[str, float]],
    output_path: str | Path,
    metric: str = "f1",
    title: str = "Per-Class F1 Score",
    dpi: int = 150,
) -> Path:
    """
    Save a bar chart of per-class metric values.

    Args:
        per_class: Dict mapping class_name -> {metric_name: value}.
        output_path: File path for the saved image.
        metric: Which metric to plot ("f1", "recall", "precision").
        title: Plot title.
        dpi: Image resolution.

    Returns:
        Path to the saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(per_class.keys())
    values = [per_class[n].get(metric, 0.0) for n in names]
    colors = ["#2ecc71" if n == "Normal" else "#e74c3c" for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.axhline(y=0.9, color="orange", linestyle="--", alpha=0.7, label="Threshold (0.90)")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Per-class {metric} bar chart saved -> {output_path}")
    return output_path
