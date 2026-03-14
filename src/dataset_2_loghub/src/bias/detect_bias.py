"""Detect data bias through slicing by system and severity.

Slices the labeled events by:
  - system: analyzes error rate and event type distribution per system
  - severity: counts across all systems
  - event_type per system: distribution

Flags a slice as BIASED if its error rate is > 2x the mean error rate
across all systems (or if a single system dominates a particular event type).

Output: data/processed/ds2_loghub/mlops_processed/bias_report.json
This task is informational — it logs warnings but does NOT fail the pipeline.
"""
import sys
from pathlib import Path

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))
from utils.paths import get_ds2_processed_dir

import polars as pl
from utils.io import read_parquet, write_json
from utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = get_ds2_processed_dir()
EVENTS_PATH = PROCESSED_DIR / "mlops_processed" / "mlops_events.parquet"
REPORT_PATH = PROCESSED_DIR / "mlops_processed" / "bias_report.json"

BIAS_THRESHOLD_MULTIPLIER = 2.0
MIN_DOMINANT_PCT = 0.80


def detect_bias(events_path: Path = EVENTS_PATH,
                report_path: Path = REPORT_PATH) -> dict:
    logger.info("Loading %s", events_path)
    df = read_parquet(str(events_path))

    total_rows = df.height
    logger.info("Total events: %d", total_rows)

    # ── Slice 1: Per-system statistics ───────────────────────────────────────
    by_system: dict = {}
    systems = df["system"].unique().sort().to_list()
    for system in systems:
        grp = df.filter(pl.col("system") == system)
        n = grp.height
        n_error = grp.filter(pl.col("severity") == "ERROR").height
        error_rate = round(n_error / n, 4) if n > 0 else 0.0

        et_counts = grp["event_type"].value_counts().sort("count", descending=True)
        top_event_type = et_counts["event_type"][0] if n > 0 else "unknown"

        et_dist = {
            row[0]: row[1]
            for row in et_counts.iter_rows()
        }

        by_system[system] = {
            "count": int(n),
            "pct_of_total": round(n / total_rows * 100, 2),
            "error_count": int(n_error),
            "error_rate": error_rate,
            "top_event_type": top_event_type,
            "event_type_distribution": et_dist,
        }

    # ── Slice 2: Severity distribution ───────────────────────────────────────
    sev_counts = df["severity"].value_counts()
    by_severity = {row[0]: int(row[1]) for row in sev_counts.iter_rows()}

    # ── Slice 3: Event type per system (pivot) ────────────────────────────────
    event_type_per_system: dict = {}
    etypes = df["event_type"].unique().sort().to_list()
    for etype in etypes:
        grp = df.filter(pl.col("event_type") == etype)
        total_etype = grp.height
        system_dist = {
            row[0]: int(row[1])
            for row in grp["system"].value_counts().iter_rows()
        }
        event_type_per_system[etype] = {
            "total": total_etype,
            "by_system": system_dist,
        }

    # ── Bias flags ────────────────────────────────────────────────────────────
    flags: list[dict] = []

    global_error_count = df.filter(pl.col("severity") == "ERROR").height
    global_error_rate = global_error_count / total_rows
    threshold = global_error_rate * BIAS_THRESHOLD_MULTIPLIER
    logger.info("Global error rate: %.3f | Bias threshold: %.3f", global_error_rate, threshold)

    for system, info in by_system.items():
        er = info["error_rate"]
        if er > threshold and threshold > 0:
            flags.append({
                "slice": system,
                "metric": "error_rate",
                "value": er,
                "global_mean": round(global_error_rate, 4),
                "threshold": round(threshold, 4),
                "status": "BIASED",
                "description": (
                    f"System '{system}' has error_rate={er:.3f}, which is "
                    f">{BIAS_THRESHOLD_MULTIPLIER}x the global mean={global_error_rate:.3f}"
                ),
            })
            logger.warning("BIAS FLAG: %s error_rate=%.3f > threshold=%.3f",
                           system, er, threshold)

    for etype, info in event_type_per_system.items():
        if info["total"] < 5:
            continue
        for system, count in info["by_system"].items():
            pct = count / info["total"]
            if pct > MIN_DOMINANT_PCT:
                flags.append({
                    "slice": f"{system}/{etype}",
                    "metric": "event_type_dominance",
                    "value": round(pct, 4),
                    "threshold": MIN_DOMINANT_PCT,
                    "status": "BIASED",
                    "description": (
                        f"System '{system}' accounts for {pct*100:.1f}% of "
                        f"event_type='{etype}' ({count}/{info['total']} events)"
                    ),
                })
                logger.warning("BIAS FLAG: %s dominates event_type '%s' (%.1f%%)",
                               system, etype, pct * 100)

    bias_detected = len(flags) > 0
    if bias_detected:
        logger.warning("Bias detected in %d slice(s). Review bias_report.json.", len(flags))
    else:
        logger.info("No significant bias detected across system slices.")

    report = {
        "bias_detected": bias_detected,
        "total_events": total_rows,
        "global_error_rate": round(global_error_rate, 4),
        "slices": {
            "by_system": by_system,
            "by_severity": by_severity,
            "by_event_type_per_system": event_type_per_system,
        },
        "flags": flags,
        "methodology": (
            "Systems flagged if error_rate > 2x global mean. "
            "Event types flagged if one system contributes >80% of occurrences."
        ),
        "mitigation_notes": (
            "If bias is detected: consider re-sampling underrepresented systems, "
            "adjusting labeling thresholds per system, or collecting more balanced data."
        ),
    }

    write_json(report, str(report_path))
    logger.info("Bias report written to %s", report_path)
    return report


if __name__ == "__main__":
    detect_bias()
