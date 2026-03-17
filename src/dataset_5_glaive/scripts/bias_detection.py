"""
Bias Detection via Data Slicing for Glaive Function Calling v2
Uses Polars for analysis, with pandas bridge for Fairlearn where needed.
"""

import json
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "glaive_processed.jsonl"
BIAS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "validation"

REPRESENTATION_THRESHOLD = 0.05


def load_data(filepath: Path) -> pl.DataFrame:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pl.DataFrame(records)


def add_slice_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.when(pl.col("num_turns") <= 1).then(pl.lit("single"))
          .when(pl.col("num_turns") <= 3).then(pl.lit("short"))
          .when(pl.col("num_turns") <= 5).then(pl.lit("medium"))
          .otherwise(pl.lit("long"))
          .alias("turn_bucket"),
        pl.when(pl.col("num_calls") <= 0).then(pl.lit("no_calls"))
          .when(pl.col("num_calls") <= 1).then(pl.lit("one_call"))
          .when(pl.col("num_calls") <= 2).then(pl.lit("two_calls"))
          .otherwise(pl.lit("many_calls"))
          .alias("call_bucket"),
        (pl.col("num_defined_functions") > 0).alias("has_defined_functions"),
    ])


def analyze_slice(df: pl.DataFrame, slice_col: str) -> pl.DataFrame:
    total = df.height
    groups = df.group_by(slice_col).agg([
        pl.len().alias("count"),
        pl.col("num_turns").mean().alias("avg_turns"),
        pl.col("num_calls").mean().alias("avg_calls"),
        pl.col("has_malformed").sum().alias("malformed_count"),
        pl.col("has_error_handling").mean().alias("error_handling_pct"),
    ])
    return groups.with_columns([
        (pl.col("count") / total).round(4).alias("proportion"),
        (pl.col("count") / total < REPRESENTATION_THRESHOLD).alias("is_underrepresented"),
    ]).rename({slice_col: "slice_value"}).with_columns([
        pl.col("slice_value").cast(pl.Utf8),
        pl.lit(slice_col).alias("slice_column"),
    ])


def detect_representation_bias(slice_df: pl.DataFrame) -> list:
    findings = []
    under = slice_df.filter(pl.col("is_underrepresented"))
    for row in under.iter_rows(named=True):
        findings.append({
            "slice_column": row["slice_column"],
            "slice_value": str(row["slice_value"]),
            "proportion": row["proportion"],
            "count": row["count"],
            "severity": "high" if row["proportion"] < 0.01 else "medium",
            "recommendation": (
                f"Slice '{row['slice_value']}' in '{row['slice_column']}' "
                f"represents only {row['proportion']:.2%} of data. "
                f"Consider oversampling or collecting more examples."
            ),
        })
    return findings


def suggest_mitigation(findings: list) -> list:
    mitigations = []
    for finding in findings:
        if finding["severity"] == "high":
            strategy = "oversample"
            detail = f"Apply SMOTE or random oversampling to '{finding['slice_value']}' slice."
        else:
            strategy = "collect_more"
            detail = f"Collect additional examples for '{finding['slice_value']}' slice."
        mitigations.append({
            "slice_column": finding["slice_column"],
            "slice_value": finding["slice_value"],
            "strategy": strategy,
            "detail": detail,
        })
    return mitigations


def run_bias_detection(filepath: Path = PROCESSED_FILE) -> dict:
    BIAS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed data...")
    df = load_data(filepath)
    logger.info("Loaded %d records", df.height)

    df = add_slice_features(df)

    slice_columns = [
        "complexity_tier", "turn_bucket", "call_bucket",
        "has_error_handling", "has_parallel", "has_defined_functions",
    ]

    all_slice_dfs = []
    slice_results = {}
    for col in slice_columns:
        logger.info("Analyzing slice: %s", col)
        sdf = analyze_slice(df, col)
        slice_results[col] = sdf
        all_slice_dfs.append(sdf)

    combined = pl.concat(all_slice_dfs)
    findings = detect_representation_bias(combined)
    mitigations = suggest_mitigation(findings)
    logger.info("Found %d bias findings", len(findings))

    report = {
        "total_records": df.height,
        "slices_analyzed": len(slice_columns),
        "findings_count": len(findings),
        "bias_detected": len(findings) > 0,
        "findings": findings,
        "mitigations": mitigations,
        "slice_statistics": {
            col: slice_results[col].to_dicts() for col in slice_columns
        },
    }

    report_path = BIAS_DIR / "bias_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Bias report saved to %s", report_path)
    return report


if __name__ == "__main__":
    report = run_bias_detection()
    print(f"\nBias detected: {report['bias_detected']}")
    print(f"Findings: {report['findings_count']}")
