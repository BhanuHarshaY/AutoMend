"""
Bias Detection & Mitigation - Track A Pipeline
Uses Fairlearn for data slicing and bias analysis.
Polars for data loading with pandas bridge for Fairlearn.
"""

import json
import logging
import sys
from pathlib import Path
from collections import Counter

import polars as pl
import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame, selection_rate, count

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds1_raw_dir, get_ds1_processed_dir, LOGS_DIR
    RAW_DIR = get_ds1_raw_dir()
    PROCESSED_DIR = get_ds1_processed_dir()
    LOG_DIR = LOGS_DIR
except ImportError:
    RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
    PROCESSED_DIR = SCRIPT_DIR.parent / "data" / "processed"
    LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "bias_detection.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

RAW_BATCH   = RAW_DIR / "batch_task_sample.csv"
SEQUENCES   = PROCESSED_DIR / "format_a_sequences.json"
REPORT_PATH = PROCESSED_DIR / "bias_report.json"

LABEL_NAMES = {0: "Normal", 1: "Resource_Exhaustion", 2: "System_Crash", 3: "Network_Failure", 4: "Data_Drift"}


def load_raw_data() -> pd.DataFrame:
    """Load raw batch task data via Polars, convert to pandas for Fairlearn."""
    df_pl = pl.read_csv(
        RAW_BATCH, has_header=False,
        new_columns=["start_time", "end_time", "inst_num", "task_type",
                      "job_id", "status", "plan_cpu", "plan_mem"],
    ).with_columns([
        pl.col("status").fill_null("Unknown"),
        pl.col("task_type").fill_null(-1).cast(pl.Int32),
    ])
    return df_pl.to_pandas()


def run_fairlearn_analysis(df: pd.DataFrame) -> dict:
    log.info("Running Fairlearn MetricFrame analysis...")
    df["is_failure"] = (df["status"] == "Failed").astype(int)
    y_true = df["is_failure"]
    y_pred = df["is_failure"]

    mf_status = MetricFrame(
        metrics={"selection_rate": selection_rate, "count": count},
        y_true=y_true, y_pred=y_pred,
        sensitive_features=df["status"],
    )

    log.info("  Fairlearn slice by STATUS:")
    for group, row in mf_status.by_group.iterrows():
        log.info("    %s: failure_rate=%.2f%%, count=%d", group, row["selection_rate"] * 100, int(row["count"]))

    status_disparity = mf_status.difference(method="between_groups")
    log.info("  Status disparity (selection_rate): %.4f", status_disparity["selection_rate"])

    if status_disparity["selection_rate"] > 0.1:
        log.warning("  BIAS DETECTED: High disparity across status groups: %.4f", status_disparity["selection_rate"])

    top_task_types = df["task_type"].value_counts().head(5).index
    df_top = df[df["task_type"].isin(top_task_types)].copy()
    if len(df_top) > 0:
        mf_task = MetricFrame(
            metrics={"selection_rate": selection_rate, "count": count},
            y_true=df_top["is_failure"], y_pred=df_top["is_failure"],
            sensitive_features=df_top["task_type"].astype(str),
        )
        log.info("  Fairlearn slice by TASK_TYPE (top 5):")
        for group, row in mf_task.by_group.iterrows():
            log.info("    task_type=%s: failure_rate=%.2f%%, count=%d", group, row["selection_rate"] * 100, int(row["count"]))

    return {
        "status_by_group": mf_status.by_group.to_dict(),
        "status_disparity": float(status_disparity["selection_rate"]),
        "bias_detected": bool(status_disparity["selection_rate"] > 0.1),
    }


def detect_slice_bias(df: pd.DataFrame) -> dict:
    log.info("Running manual data slicing analysis...")
    report = {}
    total = len(df)
    status_dist = df["status"].value_counts().to_dict()
    status_pct = {k: round(v / total * 100, 2) for k, v in status_dist.items()}
    dominant = {k: v for k, v in status_pct.items() if v > 80}
    if dominant:
        log.warning("  BIAS DETECTED: Dominant status classes: %s", dominant)
    report["status_slice"] = {
        "distribution": status_pct,
        "bias_detected": len(dominant) > 0,
        "dominant_classes": dominant,
    }
    return report


def detect_sequence_bias() -> dict:
    log.info("Checking sequence label imbalance...")
    with open(SEQUENCES, "r") as f:
        sequences = json.load(f)
    total = len(sequences)
    label_dist = Counter(s["label"] for s in sequences)
    label_pct = {LABEL_NAMES[k]: round(v / total * 100, 2) for k, v in label_dist.items()}
    normal_pct = label_pct.get("Normal", 0)
    bias_detected = normal_pct > 70
    if bias_detected:
        log.warning("  BIAS DETECTED: Normal class dominates at %.1f%%", normal_pct)
    return {"label_distribution": label_pct, "normal_dominance_pct": normal_pct, "bias_detected": bias_detected}


def document_mitigation(sequence_bias: dict) -> dict:
    return {
        "techniques_applied": [
            "Fairlearn MetricFrame used for slice-based bias analysis",
            "Undersampling of Normal class (label=0) to max 3x failure count",
            "Oversampling of minority failure classes (labels 1-4) using random replacement",
            "Random seed=42 for reproducibility",
        ],
        "after_mitigation": {
            "label_distribution": sequence_bias["label_distribution"],
            "bias_resolved": not sequence_bias["bias_detected"],
        },
    }


def run_bias_detection():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Starting Fairlearn bias detection and mitigation analysis...")

    df = load_raw_data()
    fairlearn_report = run_fairlearn_analysis(df)
    slice_report = detect_slice_bias(df)
    sequence_bias = detect_sequence_bias()
    mitigation_docs = document_mitigation(sequence_bias)

    report = {
        "fairlearn_analysis": fairlearn_report,
        "raw_data_slicing": slice_report,
        "sequence_bias": sequence_bias,
        "mitigation": mitigation_docs,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Bias report saved -> %s", REPORT_PATH)
    return report


if __name__ == "__main__":
    report = run_bias_detection()
    print(f"\nFairlearn bias detected : {report['fairlearn_analysis']['bias_detected']}")
    print(f"Sequence bias           : {report['sequence_bias']['bias_detected']}")
