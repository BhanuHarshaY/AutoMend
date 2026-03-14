"""Data quality validation for the MLOps events dataset.

Checks:
  1. Schema — all required columns present
  2. Allowed values — system, severity, event_type
  3. Null checks — event_id, event_template, message, raw_id not null/empty
  4. Sampling sanity — sampled % between 5% and 100% of total
  5. Template coverage — every event_id in events exists in templates file
  6. No duplicate EventIds in templates

Writes: data/processed/ds2_loghub/mlops_processed/validation_report.json
Exits with code 1 if any check fails (so Airflow marks task as failed).
"""
import re
import sys
from pathlib import Path

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))
from utils.paths import get_ds2_processed_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
from utils.io import read_parquet, read_csv, write_json
from utils.logger import get_logger
from src.utils.polars_validation import (
    validate_columns_present,
    validate_no_nulls,
    validate_no_empty_strings,
    validate_allowed_values,
    validate_regex_match,
    validate_row_count,
    run_validation_suite,
)

logger = get_logger(__name__)

PROCESSED_DIR = get_ds2_processed_dir()
EVENTS_PATH    = PROCESSED_DIR / "mlops_processed" / "mlops_events.parquet"
TEMPLATES_PATH = PROCESSED_DIR / "mlops_processed" / "mlops_templates.csv"
REPORT_PATH    = PROCESSED_DIR / "mlops_processed" / "validation_report.json"

TOTAL_ROWS   = 10_000
REQUIRED_COLS = [
    "system", "timestamp", "severity", "source",
    "event_id", "event_template", "message", "raw_id", "extras", "event_type",
]
ALLOWED_SYSTEMS   = ["linux", "hpc", "hdfs", "hadoop", "spark"]
ALLOWED_SEVERITIES = ["INFO", "WARN", "ERROR"]
ALLOWED_EVENT_TYPES = [
    "auth_failure", "permission_denied", "storage_unavailable",
    "data_ingestion_failed", "compute_oom", "executor_failure",
    "network_issue", "job_failed", "system_crash", "normal_ops", "unknown",
]
EVENT_ID_PATTERN = r"^E\d+$"


def validate_quality(events_path: Path = EVENTS_PATH,
                     templates_path: Path = TEMPLATES_PATH,
                     report_path: Path = REPORT_PATH) -> bool:
    report: dict = {"checks": {}, "passed": True, "errors": []}

    def fail(check_name: str, msg: str):
        report["checks"][check_name] = {"status": "FAIL", "detail": msg}
        report["errors"].append(f"{check_name}: {msg}")
        report["passed"] = False
        logger.error("[FAIL] %s: %s", check_name, msg)

    def ok(check_name: str, msg: str = ""):
        report["checks"][check_name] = {"status": "PASS", "detail": msg}
        logger.info("[PASS] %s%s", check_name, f": {msg}" if msg else "")

    logger.info("Loading data...")
    df = read_parquet(str(events_path))
    tmpl = read_csv(str(templates_path))

    logger.info("Running checks...")

    # --- 1. Schema check ---
    schema_result = validate_columns_present(df, REQUIRED_COLS)
    if schema_result["success"]:
        ok("schema", schema_result["detail"])
    else:
        fail("schema", schema_result["detail"])

    # --- 2. Allowed values ---
    for col, allowed, name in [
        ("system", ALLOWED_SYSTEMS, "allowed_systems"),
        ("severity", ALLOWED_SEVERITIES, "allowed_severities"),
        ("event_type", ALLOWED_EVENT_TYPES, "allowed_event_types"),
    ]:
        result = validate_allowed_values(df, col, allowed)
        if result["success"]:
            ok(name, result["detail"])
        else:
            fail(name, result["detail"])

    # --- 3. Null/empty checks ---
    null_cols = ["event_id", "event_template", "message", "raw_id"]
    null_results = validate_no_nulls(df, null_cols)
    empty_results = validate_no_empty_strings(df, null_cols)
    for nr, er in zip(null_results, empty_results):
        col_name = nr["check"].replace("no_nulls_", "")
        combined_ok = nr["success"] and er["success"]
        if combined_ok:
            ok(f"nulls_{col_name}", f"No nulls or empty strings in '{col_name}'")
        else:
            details = []
            if not nr["success"]:
                details.append(nr["detail"])
            if not er["success"]:
                details.append(er["detail"])
            fail(f"nulls_{col_name}", "; ".join(details))

    # --- 4. EventId pattern ---
    eid_result = validate_regex_match(df, "event_id", EVENT_ID_PATTERN)
    if eid_result["success"]:
        ok("event_id_pattern", "All EventIds match E<digits> pattern")
    else:
        fail("event_id_pattern", eid_result["detail"])

    # --- 5. Sampling sanity (5%–100%) ---
    pct = df.height / TOTAL_ROWS * 100
    if not (5 <= pct <= 100):
        fail("sampling_sanity", f"Sampled {df.height} rows = {pct:.1f}% (expected 5–100%)")
    else:
        ok("sampling_sanity", f"Sampled {df.height} rows = {pct:.1f}% of {TOTAL_ROWS}")

    # --- 6. Template coverage ---
    tmpl_ids = set(tmpl["EventId"].unique().to_list())
    event_ids = set(df["event_id"].unique().to_list())
    missing_in_templates = event_ids - tmpl_ids
    if missing_in_templates:
        fail(
            "template_coverage",
            f"{len(missing_in_templates)} EventIds in events missing from templates: "
            f"{list(missing_in_templates)[:5]}",
        )
    else:
        ok("template_coverage", f"All {len(event_ids)} EventIds covered by templates")

    # --- 7. No duplicate (EventId, system) pairs in templates ---
    dup_count = tmpl.group_by(["EventId", "system"]).len().filter(pl.col("len") > 1).height
    if dup_count > 0:
        dup_rows = (
            tmpl.group_by(["EventId", "system"])
            .len()
            .filter(pl.col("len") > 1)
            .select(["EventId", "system"])
            .head(5)
            .to_dicts()
        )
        fail("template_no_duplicates", f"Duplicate (EventId, system) pairs in templates: {dup_rows}")
    else:
        ok("template_no_duplicates", "No duplicate EventIds in templates")

    # --- Summary ---
    report["total_events"]    = df.height
    report["total_templates"] = tmpl.height
    report["sample_pct"]      = round(pct, 2)

    write_json(report, str(report_path))
    logger.info("Report written to %s", report_path)

    if report["passed"]:
        logger.info("All checks passed.")
    else:
        logger.error("%d check(s) FAILED: %s", len(report["errors"]), report["errors"])

    return report["passed"]


if __name__ == "__main__":
    passed = validate_quality()
    if not passed:
        sys.exit(1)
