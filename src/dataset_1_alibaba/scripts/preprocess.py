"""
Preprocessing Pipeline - Track A: Trigger Engine (Anomaly Classification)
Alibaba Cluster Trace 2017

Uses Polars for vectorized transformations and Ray for parallel
processing of the 3 independent input files.

Steps:
1. Load all 3 raw files with correct column names
2. Feature selection
3. Discretization (tokenization) - bin floats into token IDs for BERT
4. Sliding window - group into 5-minute windows
5. Label logic - assign labels 0-4
6. Class balancing - undersample Normal, oversample Failures
7. Output Format A: {"sequence_ids": [...], "label": int}
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import polars as pl
import ray

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds1_raw_dir, get_ds1_processed_dir, LOGS_DIR
    from src.config.ray_config import init_ray, get_dataset_config
    RAW_DIR = get_ds1_raw_dir()
    PROCESSED_DIR = get_ds1_processed_dir()
    LOG_DIR = LOGS_DIR
except ImportError:
    RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
    PROCESSED_DIR = SCRIPT_DIR.parent / "data" / "processed"
    LOG_DIR = PROJECT_ROOT / "logs"

    def init_ray(**kw):
        if not ray.is_initialized():
            ray.init(num_cpus=4, ignore_reinit_error=True)

    def get_dataset_config(_):
        return {"window_size": 5}

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "preprocess.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

SERVER_USAGE = RAW_DIR / "server_usage_sample.csv"
BATCH_TASK   = RAW_DIR / "batch_task_sample.csv"
SERVER_EVENT = RAW_DIR / "server_event_sample.csv"
OUTPUT_PATH  = PROCESSED_DIR / "format_a_sequences.json"

STATUS_TOKENS = {"Terminated": 300, "Failed": 301, "Waiting": 302, "Running": 303, "Unknown": 304}
EVENT_TOKENS  = {"add": 400, "remove": 401, "failure": 402, "unknown": 403}


# ---------------------------------------------------------------------------
# Ray remote tasks — each processes one CSV independently
# ---------------------------------------------------------------------------

@ray.remote
def process_server_usage(path: str, window_size: int) -> list[dict]:
    """Discretize server usage and create sliding-window sequences."""
    df = pl.read_csv(
        path, has_header=False,
        new_columns=["time_stamp", "machine_id", "cpu_util_percent",
                      "mem_util_percent", "net_in", "net_out",
                      "disk_io_percent", "extra"],
    ).sort("time_stamp")

    df = df.with_columns([
        (pl.col("cpu_util_percent").cast(pl.Float64) / 10)
            .floor().cast(pl.Int32).clip(0, 9).alias("cpu_token") + 100,
        (pl.col("mem_util_percent").cast(pl.Float64) / 10)
            .floor().cast(pl.Int32).clip(0, 9).alias("mem_token") + 200,
    ])

    cpu_tokens = df["cpu_token"].to_list()
    mem_tokens = df["mem_token"].to_list()
    n = len(df)
    sequences = []
    for i in range(0, n - window_size + 1, window_size):
        token_ids = []
        for j in range(i, i + window_size):
            token_ids.extend([cpu_tokens[j], mem_tokens[j]])
        last_mem = mem_tokens[i + window_size - 1]
        last_cpu = cpu_tokens[i + window_size - 1]
        label = 1 if (last_mem >= 207 or last_cpu >= 208) else 0
        sequences.append({"sequence_ids": token_ids, "label": label})
    return sequences


@ray.remote
def process_batch_task(path: str) -> list[dict]:
    """Tokenize batch task records."""
    df = pl.read_csv(
        path, has_header=False,
        new_columns=["start_time", "end_time", "inst_num", "task_type",
                      "job_id", "status", "plan_cpu", "plan_mem"],
    )

    df = df.with_columns([
        (pl.col("plan_cpu").cast(pl.Float64) / 10)
            .floor().cast(pl.Int32).clip(0, 9).alias("cpu_token") + 100,
        (pl.col("plan_mem").cast(pl.Float64) * 100 / 10)
            .floor().cast(pl.Int32).clip(0, 9).alias("mem_token") + 200,
        pl.col("status").fill_null("Unknown"),
    ])

    sequences = []
    for row in df.iter_rows(named=True):
        cpu_t = row["cpu_token"]
        mem_t = row["mem_token"]
        status = str(row["status"])
        status_t = STATUS_TOKENS.get(status, 304)

        if status == "Failed" and cpu_t <= 103:
            label = 2
        elif status == "Terminated" and mem_t >= 207:
            label = 1
        elif status in ("Failed", "Waiting"):
            label = 4
        else:
            label = 0

        sequences.append({"sequence_ids": [cpu_t, mem_t, status_t], "label": label})
    return sequences


@ray.remote
def process_server_event(path: str) -> list[dict]:
    """Tokenize server event records."""
    df = pl.read_csv(
        path, has_header=False,
        new_columns=["time_stamp", "machine_id", "event_type",
                      "event_detail", "plan_cpu", "plan_mem", "extra"],
    )

    df = df.with_columns([
        (pl.col("plan_cpu").cast(pl.Float64) / 10)
            .floor().cast(pl.Int32).clip(0, 9).alias("cpu_token") + 100,
        (pl.col("plan_mem").cast(pl.Float64) * 100 / 10)
            .floor().cast(pl.Int32).clip(0, 9).alias("mem_token") + 200,
        pl.col("event_type").fill_null("unknown").str.to_lowercase(),
    ])

    sequences = []
    for row in df.iter_rows(named=True):
        et = row["event_type"]
        event_t = EVENT_TOKENS.get(et, 403)
        label = 3 if et == "failure" else 0
        sequences.append({
            "sequence_ids": [event_t, row["cpu_token"], row["mem_token"]],
            "label": label,
        })
    return sequences


# ---------------------------------------------------------------------------
# Class balancing (Polars-backed)
# ---------------------------------------------------------------------------

def balance_classes(sequences: list[dict]) -> list[dict]:
    log.info("Balancing classes...")
    df = pl.DataFrame(sequences)
    label_counts = df.group_by("label").len().sort("label")
    log.info("  Before balancing: %s", dict(zip(
        label_counts["label"].to_list(), label_counts["len"].to_list()
    )))

    by_label = {lbl: df.filter(pl.col("label") == lbl) for lbl in range(5)}
    failure_counts = [by_label[l].height for l in range(1, 5) if by_label[l].height > 0]
    if not failure_counts:
        log.warning("  No failure samples — skipping balancing")
        return sequences

    target_failure = max(failure_counts)
    target_normal = min(by_label[0].height, target_failure * 3)

    balanced_parts = []
    normal = by_label[0]
    if normal.height > target_normal:
        normal = normal.sample(n=target_normal, seed=42)
    balanced_parts.append(normal)

    for lbl in range(1, 5):
        part = by_label[lbl]
        if part.height == 0:
            continue
        if part.height < target_failure:
            part = part.sample(n=target_failure, with_replacement=True, seed=42)
        balanced_parts.append(part)

    balanced = pl.concat(balanced_parts).sample(fraction=1.0, seed=42)
    result = balanced.to_dicts()

    label_dist = Counter(r["label"] for r in result)
    log.info("  After balancing: %s", dict(label_dist))
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_preprocessing():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cfg = get_dataset_config("ds1")

    init_ray()

    log.info("Dispatching 3 CSV files to Ray tasks...")
    refs = [
        process_server_usage.remote(str(SERVER_USAGE), cfg.get("window_size", 5)),
        process_batch_task.remote(str(BATCH_TASK)),
        process_server_event.remote(str(SERVER_EVENT)),
    ]
    results = ray.get(refs)
    sequences = []
    for r in results:
        sequences.extend(r)
    log.info("Total sequences before balancing: %d", len(sequences))

    sequences = balance_classes(sequences)
    log.info("Total sequences after balancing: %d", len(sequences))

    with open(OUTPUT_PATH, "w") as f:
        json.dump(sequences, f, indent=2)
    log.info("Saved Format A sequences -> %s", OUTPUT_PATH)

    # Also write Parquet directly for downstream consumers
    pl.DataFrame(sequences).write_parquet(
        str(PROCESSED_DIR / "format_a_sequences.parquet")
    )
    log.info("Saved Parquet -> %s", PROCESSED_DIR / "format_a_sequences.parquet")
    return sequences


if __name__ == "__main__":
    seqs = run_preprocessing()
    print(f"\nTotal sequences: {len(seqs)}")
    print(f"Label distribution: {dict(Counter(s['label'] for s in seqs))}")
