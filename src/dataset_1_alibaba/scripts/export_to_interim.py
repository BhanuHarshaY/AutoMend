"""
Export Dataset 1 (Alibaba) output to the standardized interim location.

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
    from src.config.paths import get_ds1_processed_dir, get_ds1_interim_path
    INPUT_PATH = get_ds1_processed_dir() / "format_a_sequences.json"
    OUTPUT_PATH = get_ds1_interim_path()
except ImportError:
    DATASET_DIR = SCRIPT_DIR.parent
    INPUT_PATH = DATASET_DIR / "data" / "processed" / "format_a_sequences.json"
    OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "ds1_alibaba.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def export_to_interim():
    log.info("Source: %s", INPUT_PATH)
    log.info("Destination: %s", OUTPUT_PATH)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    with open(INPUT_PATH, "r") as f:
        sequences = json.load(f)

    log.info("Loaded %d sequences from JSON", len(sequences))

    df = pl.DataFrame(sequences)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(OUTPUT_PATH))

    log.info("Exported to: %s", OUTPUT_PATH)
    log.info("File size: %.2f KB", OUTPUT_PATH.stat().st_size / 1024)
    return OUTPUT_PATH


if __name__ == "__main__":
    export_to_interim()
    print(f"Export complete: {OUTPUT_PATH}")
