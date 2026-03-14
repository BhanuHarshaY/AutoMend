"""Generate data schema validation and statistics using Polars validation.

Reads the labeled events parquet, validates schema/values using polars_validation,
and produces a combined statistics + validation report.

Output: data/processed/ds2_loghub/mlops_processed/statistics_report.json
Exits with code 1 if Polars schema validation fails.
"""
import sys
import json
from pathlib import Path

DS2_SRC = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DS2_SRC.parent.parent
sys.path.insert(0, str(DS2_SRC))
sys.path.insert(0, str(PROJECT_ROOT))
from utils.paths import get_ds2_processed_dir

import polars as pl
from utils.io import read_parquet, write_json
from utils.logger import get_logger
from src.utils.polars_validation import (
    validate_columns_present,
    validate_no_nulls,
    validate_allowed_values,
    validate_regex_match,
    validate_row_count,
)

logger = get_logger(__name__)

PROCESSED_DIR = get_ds2_processed_dir()
EVENTS_PATH = PROCESSED_DIR / "mlops_processed" / "mlops_events.parquet"
REPORT_PATH = PROCESSED_DIR / "mlops_processed" / "statistics_report.json"

REQUIRED_COLS = [
    "system", "timestamp", "severity", "source",
    "event_id", "event_template", "message", "raw_id", "extras", "event_type",
]
ALLOWED_SYSTEMS    = ["linux", "hpc", "hdfs", "hadoop", "spark"]
ALLOWED_SEVERITIES = ["INFO", "WARN", "ERROR"]
ALLOWED_EVENT_TYPES = [
    "auth_failure", "permission_denied", "storage_unavailable",
    "data_ingestion_failed", "compute_oom", "executor_failure",
    "network_issue", "job_failed", "system_crash", "normal_ops", "unknown",
]


def _run_polars_validation(df: pl.DataFrame) -> dict:
    """Run Polars validation and return result dict."""
    results = []
    all_ok = True

    r = validate_columns_present(df, REQUIRED_COLS)
    results.append({"check": "columns_present", "success": r["success"], "detail": r["detail"]})
    if not r["success"]:
        all_ok = False

    for col, allowed, name in [
        ("system", ALLOWED_SYSTEMS, "allowed_systems"),
        ("severity", ALLOWED_SEVERITIES, "allowed_severities"),
        ("event_type", ALLOWED_EVENT_TYPES, "allowed_event_types"),
    ]:
        r = validate_allowed_values(df, col, allowed)
        results.append({"check": name, "success": r["success"], "detail": r["detail"]})
        if not r["success"]:
            all_ok = False

    for col in ["event_id", "event_template", "message", "raw_id", "system"]:
        for nr in validate_no_nulls(df, [col]):
            results.append({"check": nr["check"], "success": nr["success"], "detail": nr["detail"]})
            if not nr["success"]:
                all_ok = False

    r = validate_regex_match(df, "event_id", r"^E\d+$")
    results.append({"check": "event_id_pattern", "success": r["success"], "detail": r["detail"]})
    if not r["success"]:
        all_ok = False

    r = validate_row_count(df, min_rows=1)
    results.append({"check": "row_count", "success": r["success"], "detail": r["detail"]})
    if not r["success"]:
        all_ok = False

    logger.info("Polars validation: %s (%d checks)",
                "PASSED" if all_ok else "FAILED", len(results))
    return {"success": all_ok, "results": results}


def generate_statistics(events_path: Path = EVENTS_PATH,
                        report_path: Path = REPORT_PATH) -> dict:
    logger.info("Loading %s", events_path)
    df = read_parquet(str(events_path))

    # ── Polars schema validation ─────────────────────────────────────────────
    polars_result = _run_polars_validation(df)

    # ── Polars statistics ────────────────────────────────────────────────────
    null_rates = {
        col: round(df[col].null_count() / df.height * 100, 2)
        for col in df.columns
    }

    def _vc_to_dict(series_name: str) -> dict:
        vc = df[series_name].value_counts()
        return dict(zip(vc[series_name].to_list(), vc["count"].to_list()))

    stats = {
        "total_rows": df.height,
        "total_columns": len(df.columns),
        "columns": df.columns,
        "rows_per_system": _vc_to_dict("system"),
        "severity_distribution": _vc_to_dict("severity"),
        "event_type_distribution": _vc_to_dict("event_type"),
        "null_rates_pct": null_rates,
        "unique_event_ids": df["event_id"].n_unique(),
        "unique_event_templates": df["event_template"].n_unique(),
    }

    logger.info("Total rows: %d | Systems: %s | Unique EventIds: %d",
                stats["total_rows"],
                list(stats["rows_per_system"].keys()),
                stats["unique_event_ids"])

    report = {
        "polars_validation": polars_result,
        "statistics": stats,
    }

    write_json(report, str(report_path))
    logger.info("Statistics report written to %s", report_path)

    # Fail pipeline if Polars validation found schema violations
    if polars_result["success"] is False:
        logger.error("Polars schema validation FAILED — check statistics_report.json")
        return report

    return report


if __name__ == "__main__":
    result = generate_statistics()
    if result["polars_validation"]["success"] is False:
        sys.exit(1)
