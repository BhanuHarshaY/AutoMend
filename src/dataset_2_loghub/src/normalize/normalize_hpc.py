"""Normalize HPC_2k.log_structured.csv into the unified schema.

Actual columns: LineId, LogId, Node, Component, State, Time, Flag, Content, EventId, EventTemplate
No Level column — severity from State + Flag (Flag='1' means alert/error).
Timestamp: Unix epoch integer stored as string (e.g., "1077804742").
State example: "state_change.unavailable"
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

INPUT  = RAW_DIR / "HPC" / "HPC_2k.log_structured.csv"
OUTPUT = PROCESSED_DIR / "normalized" / "hpc.parquet"

UNIFIED_COLS = [
    "system", "timestamp", "severity", "source",
    "event_id", "event_template", "message", "raw_id", "extras", "event_type",
]


def normalize_severity(state: str, flag: str, message: str) -> str:
    """Infer severity from State, Flag, and message (ERROR first, then WARN, else INFO)."""
    state_lower = (state or "").lower()
    msg_lower = (message or "").lower()
    if (flag or "").strip() == "1":
        return "ERROR"
    if "unavailable" in state_lower:
        return "ERROR"
    for kw in ("unavailable", "panic", "fatal", "error"):
        if kw in msg_lower:
            return "ERROR"
    for kw in ("degraded", "fail", "error"):
        if kw in state_lower:
            return "WARN"
    for kw in ("timeout", "retry", "slow"):
        if kw in msg_lower:
            return "WARN"
    return "INFO"


def normalize_hpc(input_path: Path = INPUT, output_path: Path = OUTPUT):
    logger.info("Reading %s", input_path)
    df = read_csv(str(input_path))

    state_lower = pl.col("State").str.to_lowercase()
    msg_lower = pl.col("Content").str.to_lowercase()

    err_expr = state_lower.str.contains("unavailable", literal=True)
    err_expr = err_expr | (pl.col("Flag").str.strip_chars() == "1")
    for kw in ("unavailable", "panic", "fatal", "error"):
        err_expr = err_expr | msg_lower.str.contains(kw, literal=True)

    warn_expr = pl.lit(False)
    for kw in ("degraded", "fail", "error"):
        warn_expr = warn_expr | state_lower.str.contains(kw, literal=True)
    for kw in ("timeout", "retry", "slow"):
        warn_expr = warn_expr | msg_lower.str.contains(kw, literal=True)

    severity = pl.when(err_expr).then(pl.lit("ERROR")).when(warn_expr).then(pl.lit("WARN")).otherwise(pl.lit("INFO"))

    raw_id = pl.when(pl.col("LogId").str.strip_chars() != "").then(pl.col("LogId")).otherwise(pl.col("LineId"))
    source = pl.concat_str([pl.col("Node"), pl.col("Component")], separator="|")

    out = df.with_columns([
        raw_id.alias("raw_id"),
        pl.col("Time").alias("timestamp"),
        source.alias("source"),
        pl.lit("hpc").alias("system"),
        pl.col("Content").alias("message"),
        pl.col("EventId").alias("event_id"),
        pl.col("EventTemplate").alias("event_template"),
        severity.alias("severity"),
        pl.struct([
            pl.col("Node").alias("node"),
            pl.col("Component").alias("component"),
            pl.col("State").alias("state"),
            pl.col("Flag").alias("flag"),
        ]).map_elements(lambda s: json.dumps(dict(s)), return_dtype=pl.Utf8).alias("extras"),
        pl.lit("").alias("event_type"),
    ]).select(UNIFIED_COLS)

    write_parquet(out, str(output_path))
    logger.info("Done — %d rows.", out.height)
    return out


if __name__ == "__main__":
    normalize_hpc()
