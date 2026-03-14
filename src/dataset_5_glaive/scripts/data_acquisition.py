"""
Data Acquisition Script for Glaive Function Calling v2
Streams dataset from HuggingFace via Ray Data and saves a reproducible sample.
"""

import json
import logging
import os
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

DATASET_NAME = "glaiveai/glaive-function-calling-v2"

try:
    from src.config.paths import get_ds5_raw_dir
    from src.config.ray_config import init_ray, get_dataset_config
    RAW_DIR = get_ds5_raw_dir()
except ImportError:
    RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

    def init_ray(**kw):
        pass

    def get_dataset_config(_):
        return {"sample_size": 5000}

OUTPUT_FILE = RAW_DIR / "glaive_raw.jsonl"


def fetch_and_save(
    dataset_name: str = DATASET_NAME,
    sample_size: int | None = None,
    output_file: Path = OUTPUT_FILE,
) -> int:
    """
    Stream dataset from HuggingFace, optionally using Ray Data for
    distributed download + parse. Falls back to sequential collection.
    Returns number of records saved.
    """
    cfg = get_dataset_config("ds5")
    if sample_size is None:
        sample_size = cfg.get("sample_size", 5000)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Starting streaming from HuggingFace: %s", dataset_name)

    try:
        import ray
        init_ray()
        hf_ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
        ds = ray.data.from_huggingface(hf_ds).limit(sample_size)
        records = ds.take_all()
        logger.info("Collected %d records via Ray Data", len(records))
    except Exception as e:
        logger.warning("Ray Data path failed (%s), falling back to sequential", e)
        hf_ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
        records = []
        for i, record in enumerate(hf_ds):
            if i >= sample_size:
                break
            records.append(record)
            if (i + 1) % 500 == 0:
                logger.info("  Collected %d / %d records", i + 1, sample_size)

    logger.info("Saving %d records to %s", len(records), output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Data acquisition complete. Records saved: %d", len(records))
    return len(records)


if __name__ == "__main__":
    count = fetch_and_save()
    print(f"\nSuccessfully saved {count} records to {OUTPUT_FILE}")
