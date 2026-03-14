"""Normalize Linux_2k.log_structured.csv into the unified schema.

Actual columns: LineId, Month, Date, Time, Level, Component, PID, Content, EventId, EventTemplate
Level is always 'combo' (non-standard) → severity inferred from message keywords.
Timestamp: "Jun 14 15:16:01"
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

# Use centralized paths, fallback to legacy
RAW_DIR = get_ds2_raw_dir() / "loghub"
if not RAW_DIR.exists():
    RAW_DIR = get_legacy_raw_dir() / "loghub"

PROCESSED_DIR = get_ds2_processed_dir()

INPUT  = RAW_DIR / "Linux" / "Linux_2k.log_structured.csv"
OUTPUT = PROCESSED_DIR / "normalized" / "linux.parquet"

# Applied in order — first match wins
ERROR_KEYWORDS = [
    "authentication failure", "failed password", "permission denied",
    "not permitted", "denied", "invalid", "refused", "error",
    "failure", "segfault", "panic",
]
WARN_KEYWORDS = [
    "warn", "timeout", "timed out", "retry", "unable",
    "deprecated", "slow", "backoff",
]

UNIFIED_COLS = [
    "system", "timestamp", "severity", "source",
    "event_id", "event_template", "message", "raw_id", "extras", "event_type",
]


def normalize_severity(message: str) -> str:
    """Infer severity from message keywords (ERROR first, then WARN, else INFO)."""
    msg_lower = (message or "").lower()
    for kw in ERROR_KEYWORDS:
        if kw in msg_lower:
            return "ERROR"
    for kw in WARN_KEYWORDS:
        if kw in msg_lower:
            return "WARN"
    return "INFO"


def normalize_linux(input_path: Path = INPUT, output_path: Path = OUTPUT):
    logger.info("Reading %s", input_path)
    df = read_csv(str(input_path))

    message_lower = pl.col("Content").str.to_lowercase()
    err_expr = pl.lit(False)
    for kw in ERROR_KEYWORDS:
        err_expr = err_expr | message_lower.str.contains(kw, literal=True)
    warn_expr = pl.lit(False)
    for kw in WARN_KEYWORDS:
        warn_expr = warn_expr | message_lower.str.contains(kw, literal=True)
    severity = pl.when(err_expr).then(pl.lit("ERROR")).when(warn_expr).then(pl.lit("WARN")).otherwise(pl.lit("INFO"))

    out = df.with_columns([
        pl.concat_str([pl.col("Month"), pl.col("Date"), pl.col("Time")], separator=" ").alias("timestamp"),
        pl.lit("linux").alias("system"),
        pl.col("LineId").alias("raw_id"),
        pl.col("Component").alias("source"),
        pl.col("Content").alias("message"),
        pl.col("EventId").alias("event_id"),
        pl.col("EventTemplate").alias("event_template"),
        severity.alias("severity"),
        pl.struct([pl.col("PID").alias("pid"), pl.col("Level").alias("level_raw")])
            .map_elements(lambda s: json.dumps(dict(s)), return_dtype=pl.Utf8)
            .alias("extras"),
        pl.lit("").alias("event_type"),
    ]).select(UNIFIED_COLS)

    write_parquet(out, str(output_path))
    logger.info("Done — %d rows.", out.height)
    return out


if __name__ == "__main__":
    normalize_linux()
