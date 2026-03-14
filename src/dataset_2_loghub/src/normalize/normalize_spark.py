"""Normalize Spark_2k.log_structured.csv into the unified schema.

Actual columns: LineId, Date, Time, Level, Component, Content, EventId, EventTemplate
Level is standard (INFO, WARN, ERROR, FATAL) — direct mapping.
Timestamp: "17/06/09 20:10:40" (YY/MM/DD HH:MM:SS) — kept as string.
"""
import json
import sys
from pathlib import Path

import polars as pl

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))
from utils.paths import get_ds2_raw_dir, get_ds2_processed_dir, get_legacy_raw_dir

from utils.io import read_csv, write_parquet
from utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = get_ds2_raw_dir() / "loghub"
if not RAW_DIR.exists():
    RAW_DIR = get_legacy_raw_dir() / "loghub"

PROCESSED_DIR = get_ds2_processed_dir()

INPUT  = RAW_DIR / "Spark" / "Spark_2k.log_structured.csv"
OUTPUT = PROCESSED_DIR / "normalized" / "spark.parquet"

LEVEL_MAP = {
    "INFO":    "INFO",
    "WARN":    "WARN",
    "WARNING": "WARN",
    "ERROR":   "ERROR",
    "FATAL":   "ERROR",
    "DEBUG":   "INFO",
    "TRACE":   "INFO",
}
ERROR_KW = ["exception", "failed", "failure", "error", "killed", "oom"]
WARN_KW  = ["warn", "timeout", "retry", "slow"]

UNIFIED_COLS = [
    "system", "timestamp", "severity", "source",
    "event_id", "event_template", "message", "raw_id", "extras", "event_type",
]


def normalize_severity(level: str, message: str) -> str:
    """Infer severity from Level (if mapped) or message keywords."""
    level_upper = (level or "").strip().upper()
    if level_upper in LEVEL_MAP:
        return LEVEL_MAP[level_upper]
    msg_lower = (message or "").lower()
    for kw in ERROR_KW:
        if kw in msg_lower:
            return "ERROR"
    for kw in WARN_KW:
        if kw in msg_lower:
            return "WARN"
    return "INFO"


def _level_mapped_severity():
    """Build severity from LEVEL_MAP: when Level matches key, use value."""
    expr = pl.lit(None).cast(pl.Utf8)
    for raw, mapped in LEVEL_MAP.items():
        expr = pl.when(pl.col("Level").str.strip_chars().str.to_uppercase() == raw).then(pl.lit(mapped)).otherwise(expr)
    return expr


def normalize_spark(input_path: Path = INPUT, output_path: Path = OUTPUT):
    logger.info("Reading %s", input_path)
    df = read_csv(str(input_path))

    msg_lower = pl.col("Content").str.to_lowercase()
    level_mapped = _level_mapped_severity()

    err_expr = pl.lit(False)
    for kw in ERROR_KW:
        err_expr = err_expr | msg_lower.str.contains(kw, literal=True)
    warn_expr = pl.lit(False)
    for kw in WARN_KW:
        warn_expr = warn_expr | msg_lower.str.contains(kw, literal=True)

    severity = (
        pl.when(level_mapped.is_not_null())
        .then(level_mapped)
        .when(err_expr)
        .then(pl.lit("ERROR"))
        .when(warn_expr)
        .then(pl.lit("WARN"))
        .otherwise(pl.lit("INFO"))
    )

    out = df.with_columns([
        pl.concat_str([pl.col("Date"), pl.col("Time")], separator=" ").alias("timestamp"),
        pl.lit("spark").alias("system"),
        pl.col("LineId").alias("raw_id"),
        pl.col("Component").alias("source"),
        pl.col("Content").alias("message"),
        pl.col("EventId").alias("event_id"),
        pl.col("EventTemplate").alias("event_template"),
        severity.alias("severity"),
        pl.struct([pl.col("Level").alias("level_raw")])
            .map_elements(lambda s: json.dumps(dict(s)), return_dtype=pl.Utf8)
            .alias("extras"),
        pl.lit("").alias("event_type"),
    ]).select(UNIFIED_COLS)

    write_parquet(out, str(output_path))
    logger.info("Done — %d rows.", out.height)
    return out


if __name__ == "__main__":
    normalize_spark()
