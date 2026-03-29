"""
metrics.py

Compute classification metrics for Model 1 anomaly detection.
Primary metrics: Macro F1-score and per-class Recall.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)
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


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_labels: int = 7,
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Returns a dict with scalar metrics and nested per-class breakdowns.
    """
    labels = list(range(num_labels))
    names = LABEL_NAMES[:num_labels]

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    anomaly_mask = y_true > 0
    anomaly_recall = float(recall_score(
        y_true[anomaly_mask], y_pred[anomaly_mask],
        average="macro", labels=list(range(1, num_labels)), zero_division=0,
    )) if anomaly_mask.sum() > 0 else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report_str = classification_report(
        y_true, y_pred, labels=labels, target_names=names, zero_division=0,
    )

    per_class = {}
    for i, name in enumerate(names):
        per_class[name] = {
            "f1": float(per_class_f1[i]),
            "recall": float(per_class_recall[i]),
            "precision": float(per_class_precision[i]),
        }

    result = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "anomaly_recall": anomaly_recall,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str,
    }

    logger.info(f"Macro F1: {macro_f1:.4f} | Accuracy: {accuracy:.4f} | Anomaly Recall: {anomaly_recall:.4f}")
    return result
