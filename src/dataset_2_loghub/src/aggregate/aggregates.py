"""Generate trigger-ready aggregate CSVs from the labeled events dataset.

Outputs:
  data/processed/ds2_loghub/mlops_processed/event_counts_by_window.csv
  data/processed/ds2_loghub/mlops_processed/error_rate_by_system.csv
  data/processed/ds2_loghub/mlops_processed/top_templates.csv
"""
import sys
from pathlib import Path

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))
from utils.paths import get_ds2_processed_dir

import polars as pl
from utils.io import read_parquet, write_csv
from utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = get_ds2_processed_dir()
EVENTS_PATH  = PROCESSED_DIR / "mlops_processed" / "mlops_events.parquet"
OUT_DIR      = PROCESSED_DIR / "mlops_processed"
TOP_N        = 10


def make_time_bucket(timestamp: str, system: str) -> str:
    """Create a pseudo time bucket from the timestamp string.

    - Hadoop:  "2015-10-18 18:01:47,978" → "2015-10-18 18:01" (minute bucket)
    - Spark:   "17/06/09 20:10:40"       → "17/06/09 20:10"
    - HDFS:    "081109 203615"            → "081109 2036" (first 4 digits of time)
    - Linux:   "Jun 14 15:16:01"         → "Jun 14 15:16"
    - HPC:     "1077804742"              → use first 7 chars as pseudo-bucket
    """
    ts = str(timestamp).strip()
    if system == "hpc":
        return ts[:7] if len(ts) >= 7 else ts
    if system == "hdfs":
        parts = ts.split()
        if len(parts) == 2:
            return parts[0] + " " + parts[1][:4]
        return ts[:9]
    if " " in ts:
        date_part, time_part = ts.rsplit(" ", 1)
        time_part = time_part.replace(",", ".")
        minute = ":".join(time_part.split(":")[:2])
        return date_part + " " + minute
    return ts[:10]


def aggregate_metrics(events_path: Path = EVENTS_PATH, out_dir: Path = OUT_DIR):
    logger.info("Reading %s", events_path)
    df = read_parquet(str(events_path))

    # --- 1. Counts by time window ---
    df = df.with_columns(
        pl.struct(["timestamp", "system"])
        .map_elements(
            lambda r: make_time_bucket(r["timestamp"], r["system"]),
            return_dtype=pl.Utf8,
        )
        .alias("time_bucket")
    )
    counts = (
        df.group_by(["system", "severity", "event_type", "time_bucket"])
        .len()
        .rename({"len": "count"})
        .sort(["system", "time_bucket", "severity"])
    )
    write_csv(counts, str(out_dir / "event_counts_by_window.csv"))
    logger.info("event_counts_by_window: %d rows", counts.height)

    # --- 2. Error rate by system ---
    totals = (
        df.group_by(["system", "time_bucket"])
        .len()
        .rename({"len": "total"})
    )
    errors = (
        df.filter(pl.col("severity") == "ERROR")
        .group_by(["system", "time_bucket"])
        .len()
        .rename({"len": "error_count"})
    )
    error_rate = totals.join(errors, on=["system", "time_bucket"], how="left")
    error_rate = error_rate.with_columns(
        pl.col("error_count").fill_null(0).cast(pl.Int64),
    ).with_columns(
        (pl.col("error_count") / pl.col("total")).round(4).alias("error_rate"),
    ).sort(["system", "time_bucket"])
    write_csv(error_rate, str(out_dir / "error_rate_by_system.csv"))
    logger.info("error_rate_by_system: %d rows", error_rate.height)

    # --- 3. Top N event templates per system ---
    template_counts = (
        df.group_by(["system", "event_id", "event_template"])
        .len()
        .rename({"len": "count"})
        .sort(["system", "count"], descending=[False, True])
    )
    top_templates = (
        template_counts
        .group_by("system", maintain_order=True)
        .head(TOP_N)
    )
    write_csv(top_templates, str(out_dir / "top_templates.csv"))
    logger.info("top_templates: %d rows", top_templates.height)

    logger.info("Done.")
    return counts, error_rate, top_templates


if __name__ == "__main__":
    aggregate_metrics()
