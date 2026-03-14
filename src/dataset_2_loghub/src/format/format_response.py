"""Convert mlops_events.parquet to Format A for log anomaly detection.

Reads:  data/processed/ds2_loghub/mlops_processed/mlops_events.parquet
Output: data/processed/ds2_loghub/data_ready/event_sequences.parquet

Format A schema (one row per 5-minute window):
    {"sequence_ids": [List[int]], "label": int}

Labels:
    0 = Normal
    1 = Resource_Exhaustion
    2 = System_Crash
    3 = Network_Failure       (keyword: Timeout, Unreachable)
    4 = Data_Drift/Corruption (keyword: Checksum, Verification Failed)
    5 = Auth_Failure
    6 = Permission_Denied

Sequence rules:
    - Max length: 512 tokens
    - Truncation: keep most recent logs (drop from start)
    - Padding: pad shorter sequences with 0s
"""
import sys
from pathlib import Path

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))
from utils.paths import get_ds2_processed_dir

import polars as pl
from utils.io import read_parquet, write_parquet
from utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = get_ds2_processed_dir()
EVENTS_PATH = PROCESSED_DIR / "mlops_processed" / "mlops_events.parquet"
OUTPUT      = PROCESSED_DIR / "data_ready" / "event_sequences.parquet"

MAX_LEN = 512
WINDOW  = "1m"

LABEL_MAP = {
    "normal_ops":            0,
    "unknown":               0,
    "compute_oom":           1,
    "storage_unavailable":   2,
    "executor_failure":      2,
    "system_crash":          2,
    "job_failed":            2,
    "network_issue":         3,
    "data_ingestion_failed": 4,
    "auth_failure":          5,
    "permission_denied":     6,
}

KEYWORD_RULES = [
    (3, ["timeout", "unreachable"]),
    (4, ["checksum", "verification failed"]),
]


def keyword_label(template: str):
    """Return event label if template matches document keyword rules, else None."""
    t = str(template).lower()
    for label, keywords in KEYWORD_RULES:
        if any(kw in t for kw in keywords):
            return label
    return None


def pad_or_truncate(seq: list, max_len: int = MAX_LEN) -> list:
    """Truncate from start (keep most recent) or pad end with zeros."""
    if len(seq) > max_len:
        return seq[-max_len:]
    return seq + [0] * (max_len - len(seq))


def format_response(events_path: Path = EVENTS_PATH, output_path: Path = OUTPUT):
    logger.info("Loading %s", events_path)
    df = read_parquet(str(events_path))

    # 1. Convert EventId string → integer (E55 → 55)
    df = df.with_columns(
        pl.col("event_id").str.extract(r"E(\d+)", 1).cast(pl.Int32).alias("event_int")
    )
    logger.info(
        "EventId range: E%d – E%d",
        df["event_int"].min(),
        df["event_int"].max(),
    )

    # 2. Assign event label from existing event_type labels
    df = df.with_columns(
        pl.col("event_type")
        .replace_strict(LABEL_MAP, default=0, return_dtype=pl.Int32)
        .alias("event_label")
    )

    # 3. Override with document keyword rules on event_template
    kw_series = df["event_template"].map_elements(keyword_label, return_dtype=pl.Int32)
    df = df.with_columns(
        pl.when(kw_series.is_not_null())
        .then(kw_series)
        .otherwise(pl.col("event_label"))
        .alias("event_label")
    )
    kw_override_count = kw_series.drop_nulls().len()
    logger.info("Keyword rule override applied to %d rows", kw_override_count)

    # 4. Parse timestamps — handle multiple formats across systems
    orig = df["timestamp"].cast(pl.Utf8)

    numeric_ts = orig.str.strip_chars().str.extract(r"^(\d+)$", 1)
    is_numeric = numeric_ts.is_not_null()

    linux_mask = orig.str.contains(r"^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}$")

    parsed_standard = orig.str.to_datetime(strict=False, format=None)
    parsed_epoch = numeric_ts.cast(pl.Int64, strict=False).cast(pl.Datetime("us"))
    parsed_linux = orig.str.to_datetime(format="%b %d %H:%M:%S", strict=False)

    ts_col = (
        pl.when(is_numeric).then(parsed_epoch)
        .when(linux_mask).then(parsed_linux)
        .otherwise(parsed_standard)
    )
    df = df.with_columns(ts_col.alias("timestamp_dt"))

    standard_count = df.filter(~is_numeric & ~linux_mask & pl.col("timestamp_dt").is_not_null()).height
    epoch_count = df.filter(is_numeric).height
    linux_count = df.filter(linux_mask & ~is_numeric).height
    nat_count = df.filter(pl.col("timestamp_dt").is_null()).height
    logger.info(
        "Timestamp parse results — standard: %d, epoch: %d, linux-fmt: %d, NaT: %d",
        standard_count, epoch_count, linux_count, nat_count,
    )

    df = df.sort(["system", "timestamp_dt"])
    df = df.with_columns(
        pl.col("timestamp_dt").dt.truncate(WINDOW).alias("window")
    )

    # 5. Build sequences grouped by (system, window)
    sequences = (
        df.group_by(["system", "window"])
        .agg(
            pl.col("event_int").alias("sequence_ids"),
            pl.col("event_label").max().alias("label"),
        )
    )

    sequences = sequences.with_columns(
        pl.col("sequence_ids")
        .map_elements(pad_or_truncate, return_dtype=pl.List(pl.Int32))
    )

    logger.info("Total sequences: %d", sequences.height)
    label_dist = sequences["label"].value_counts()
    for row in label_dist.sort("label").iter_rows():
        logger.info("  label %d: %d", row[0], row[1])

    output = sequences.select(["sequence_ids", "label"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(output, str(output_path))
    logger.info("Written %d sequences → %s", output.height, output_path)
    return output


if __name__ == "__main__":
    format_response()
