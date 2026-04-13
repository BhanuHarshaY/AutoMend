"""
Centralized data acquisition mode configuration.

Reads PIPELINE_DATA_MODE from environment and provides per-dataset
acquisition parameters for each mode (dummy / sample / full).

dummy  - Offline synthetic data, no network required. Fast, for E2E testing.
sample - Capped download from real source. Network required.
full   - Full dataset download. Network + API tokens may be required.
"""

import os
import logging
from typing import Literal

logger = logging.getLogger(__name__)

DataMode = Literal["dummy", "sample", "full"]

VALID_MODES = ("dummy", "sample", "full")


def get_data_mode() -> DataMode:
    mode = os.environ.get("PIPELINE_DATA_MODE", "dummy").strip().lower()
    if mode not in VALID_MODES:
        logger.warning(
            "Invalid PIPELINE_DATA_MODE='%s', falling back to 'dummy'. "
            "Valid values: %s",
            mode,
            ", ".join(VALID_MODES),
        )
        return "dummy"
    return mode  # type: ignore[return-value]


def is_dummy() -> bool:
    return get_data_mode() == "dummy"


def is_sample() -> bool:
    return get_data_mode() == "sample"


def is_full() -> bool:
    return get_data_mode() == "full"


# Per-dataset acquisition configs keyed by (dataset_key, mode).
# Each entry defines the parameters passed to the dataset's acquire function.
ACQUIRE_CONFIG = {
    "ds1": {
        "dummy":  {"num_rows": 100},
        "sample": {"num_rows": 1000},
        "full":   {"num_rows": None},  # None = manual placement required
    },
    "ds2": {
        "dummy":  {"source": "seed", "num_rows": 50},
        "sample": {"source": "github"},
        "full":   {"source": "github"},  # LogHub's public data is 2K per system max
    },
    "ds3": {
        "dummy":  {"source": "seed", "num_rows": 50},
        "sample": {"source": "api", "max_questions": 500},
        "full":   {"source": "api", "max_questions": 6000},
    },
    "ds4": {
        "dummy":  {"prompt_set": "default", "prompt_count": 15},
        "sample": {"prompt_set": "default", "prompt_count": 15},
        "full":   {"prompt_set": "expanded", "prompt_count": 100},
    },
    "ds5": {
        "dummy":  {"source": "seed", "num_records": 100},
        "sample": {"source": "huggingface", "sample_size": 5000},
        "full":   {"source": "huggingface", "sample_size": 6000},
    },
    "ds6": {
        "dummy":  {"source": "seed", "num_records": 50},
        "sample": {"source": "huggingface", "sample_size": 20000},
        "full":   {"source": "huggingface", "sample_size": 6000},
    },
}


def get_acquire_config(dataset_key: str) -> dict:
    """Return acquisition config dict for the given dataset and current mode."""
    mode = get_data_mode()
    try:
        cfg = ACQUIRE_CONFIG[dataset_key][mode]
    except KeyError:
        raise ValueError(
            f"No acquisition config for dataset_key='{dataset_key}', mode='{mode}'"
        )
    logger.info("Acquire config for %s [%s]: %s", dataset_key, mode, cfg)
    return cfg
