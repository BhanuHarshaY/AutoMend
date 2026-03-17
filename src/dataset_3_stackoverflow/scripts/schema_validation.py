"""
Schema Validation Module (Polars-native)
=========================================
Automated data schema and statistics generation using Polars.
Validates data quality over time and generates data documentation.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import polars as pl
import polars.selectors as cs

from config import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.polars_validation import (
    validate_columns_present,
    validate_no_nulls,
    validate_row_count,
    validate_value_range,
    validate_unique,
    run_validation_suite,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "schema_validation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

EXPECTED_SCHEMA = {
    "required_columns": [
        "question_id", "title", "question_body", "answer_body",
        "tags", "score", "quality_score",
    ],
    "not_null_columns": ["question_id", "question_body", "answer_body", "score"],
    "unique_columns": ["question_id"],
    "numeric_ranges": {
        "score": (0, None),
        "quality_score": (0, None),
    },
    "row_count": {"min": config.MIN_ROWS, "max": 1000000},
}


def generate_column_statistics(df: pl.DataFrame, column: str) -> dict:
    stats = {
        "name": column,
        "dtype": str(df[column].dtype),
        "count": df[column].drop_nulls().len(),
        "null_count": df[column].null_count(),
        "null_percentage": round(df[column].null_count() / df.height * 100, 2) if df.height > 0 else 0,
    }

    if df[column].dtype in (pl.Int64, pl.Int32, pl.Float64, pl.Float32):
        stats.update({
            "mean": df[column].mean(),
            "std": df[column].std(),
            "min": df[column].min(),
            "max": df[column].max(),
            "median": df[column].median(),
            "q25": df[column].quantile(0.25),
            "q75": df[column].quantile(0.75),
        })
    elif df[column].dtype == pl.Utf8:
        non_null = df.filter(pl.col(column).is_not_null())
        if non_null.height > 0:
            lengths = non_null[column].str.len_bytes()
            stats.update({
                "unique_count": non_null[column].n_unique(),
                "min_length": lengths.min(),
                "max_length": lengths.max(),
                "mean_length": round(lengths.mean(), 2),
            })

    return stats


def generate_data_statistics(df: pl.DataFrame) -> dict:
    logger.info("Generating data statistics...")
    stats = {
        "generated_at": datetime.now().isoformat(),
        "dataset_info": {
            "row_count": df.height,
            "column_count": len(df.columns),
            "columns": df.columns,
        },
        "column_statistics": {},
    }
    for col in df.columns:
        try:
            stats["column_statistics"][col] = generate_column_statistics(df, col)
        except Exception as e:
            logger.warning("Failed to generate stats for column %s: %s", col, e)
            stats["column_statistics"][col] = {"error": str(e)}
    return stats


def run_schema_validation() -> dict:
    logger.info("=" * 60)
    logger.info("STARTING SCHEMA VALIDATION")
    logger.info("=" * 60)

    start_time = datetime.now()

    try:
        input_path = config.processed_dir / "qa_pairs_processed.json"
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pl.DataFrame(data)
        logger.info("Loaded %d records", df.height)

        statistics = generate_data_statistics(df)
        stats_path = config.REPORTS_DIR / "statistics" / "data_statistics.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(statistics, f, indent=2, default=str)

        checks = [
            validate_columns_present(df, EXPECTED_SCHEMA["required_columns"]),
            validate_no_nulls(df, EXPECTED_SCHEMA["not_null_columns"]),
            validate_row_count(df, min_rows=EXPECTED_SCHEMA["row_count"]["min"],
                               max_rows=EXPECTED_SCHEMA["row_count"]["max"]),
        ]
        for col in EXPECTED_SCHEMA["unique_columns"]:
            checks.append(validate_unique(df, col))
        for col, (min_v, max_v) in EXPECTED_SCHEMA["numeric_ranges"].items():
            checks.append(validate_value_range(df, col, min_val=min_v, max_val=max_v))

        report = run_validation_suite(checks)

        validation_path = config.REPORTS_DIR / "validation" / "schema_validation.json"
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        metrics = {
            "schema_valid": report["all_passed"],
            "checks_passed": report["passed"],
            "checks_failed": report["failed"],
            "row_count": df.height,
            "column_count": len(df.columns),
            "timestamp": datetime.now().isoformat(),
        }
        metrics_path = config.REPORTS_DIR / "validation" / "validation_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Schema validation: %s", "PASSED" if report["all_passed"] else "FAILED")

        return {
            "status": "success",
            "valid": report["all_passed"],
            "statistics": statistics,
            "validation": report,
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
        }

    except Exception as e:
        logger.error("Schema validation failed: %s", e)
        raise


if __name__ == "__main__":
    run_schema_validation()
