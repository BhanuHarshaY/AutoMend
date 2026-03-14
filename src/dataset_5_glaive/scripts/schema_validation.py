"""
Schema Validation for Glaive Function Calling v2
Uses Polars-native validation (replaces Great Expectations).
"""

import json
import logging
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.polars_validation import (
    validate_columns_present,
    validate_no_nulls,
    validate_allowed_values,
    validate_value_range,
    validate_row_count,
    run_validation_suite,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "glaive_processed.jsonl"
VALIDATION_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "validation"

GLAIVE_SCHEMA = {
    "required_columns": [
        "system", "chat", "num_turns", "num_calls",
        "complexity_tier", "has_parallel", "has_malformed",
        "function_calls", "num_defined_functions",
        "defined_function_names", "function_signatures",
        "has_error_handling", "has_function_error_response",
        "has_conditional_error", "error_keywords_found",
    ],
    "not_null_columns": ["chat", "num_turns", "num_calls", "complexity_tier"],
    "column_value_sets": {
        "complexity_tier": ["none", "simple", "moderate", "complex", "malformed"],
    },
    "numeric_ranges": {
        "num_turns": (0, 50),
        "num_calls": (0, 20),
        "num_defined_functions": (0, 20),
    },
    "row_count_range": (4000, 6000),
}


def load_processed_data(filepath: Path) -> pl.DataFrame:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pl.DataFrame(records)


def run_validation(df: pl.DataFrame) -> dict:
    checks = [
        validate_columns_present(df, GLAIVE_SCHEMA["required_columns"]),
        validate_no_nulls(df, GLAIVE_SCHEMA["not_null_columns"]),
        validate_allowed_values(df, "complexity_tier", GLAIVE_SCHEMA["column_value_sets"]["complexity_tier"]),
    ]

    for col, (min_val, max_val) in GLAIVE_SCHEMA["numeric_ranges"].items():
        checks.append(validate_value_range(df, col, min_val=min_val, max_val=max_val))

    min_rows, max_rows = GLAIVE_SCHEMA["row_count_range"]
    checks.append(validate_row_count(df, min_rows=min_rows, max_rows=max_rows))

    if "chat" in df.columns:
        short_chats = df.filter(pl.col("chat").str.len_bytes() < 10).height
        checks.append({
            "success": short_chats == 0,
            "check": "chat_not_empty",
            "detail": f"{short_chats} chats shorter than 10 chars" if short_chats > 0 else "OK",
        })

    return run_validation_suite(checks)


def print_validation_report(report: dict) -> None:
    passed = report["passed"]
    failed = report["failed"]
    total = report["total"]
    print("\n" + "=" * 55)
    print("          DATA VALIDATION REPORT")
    print("=" * 55)
    failures = [r for r in report["results"] if not r["success"]]
    if failures:
        print("\nFAILED CHECKS:")
        for r in failures:
            print(f"   - {r['check']}: {r['detail']}")
    print(f"\n Passed: {passed}/{total}")
    print(f" Failed: {failed}/{total}")
    print(f" Success Rate: {passed / total * 100:.1f}%")
    print("=" * 55)


def save_validation_report(report: dict) -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    report_path = VALIDATION_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Validation report saved to %s", report_path)


if __name__ == "__main__":
    logger.info("Loading processed data...")
    df = load_processed_data(PROCESSED_FILE)
    logger.info("Loaded %d records", df.height)

    report = run_validation(df)
    print_validation_report(report)
    save_validation_report(report)
