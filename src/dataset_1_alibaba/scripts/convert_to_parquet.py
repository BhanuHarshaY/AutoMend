"""
Convert Dataset 1 (Alibaba) JSON output to Parquet format for Track A combiner.

Uses Polars for native Parquet write (no pandas dependency).

Reads:  data/processed/ds1_alibaba/format_a_sequences.json
Output: data/interim/ds1_alibaba.parquet
"""

import json
import logging
import sys
from pathlib import Path

import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds1_processed_dir, get_ds1_interim_path, LOGS_DIR
    INPUT_PATH = get_ds1_processed_dir() / "format_a_sequences.json"
    OUTPUT_PATH = get_ds1_interim_path()
except ImportError:
    INPUT_PATH = SCRIPT_DIR.parent / "data" / "processed" / "format_a_sequences.json"
    OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "ds1_alibaba.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def convert_to_parquet():
    log.info("Reading JSON from: %s", INPUT_PATH)
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    with open(INPUT_PATH, "r") as f:
        sequences = json.load(f)

    log.info("Loaded %d sequences", len(sequences))

    df = pl.DataFrame(sequences)
    log.info("DataFrame shape: %s", df.shape)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(OUTPUT_PATH))

    log.info("Saved Parquet to: %s", OUTPUT_PATH)
    log.info("File size: %.2f KB", OUTPUT_PATH.stat().st_size / 1024)
    return df


if __name__ == "__main__":
    df = convert_to_parquet()
    print(f"\nConversion complete! Total rows: {df.height}")
