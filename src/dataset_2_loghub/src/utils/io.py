"""I/O helpers for reading/writing pipeline data (Polars-based)."""
import sys
import json
from pathlib import Path

import polars as pl

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))

from utils.logger import get_logger

logger = get_logger(__name__)


def read_csv(path: str, **kwargs) -> pl.DataFrame:
    """Read a CSV with all columns as strings by default."""
    return pl.read_csv(path, infer_schema_length=0, **kwargs)


def write_parquet(df: pl.DataFrame, path: str) -> None:
    """Write a DataFrame to Parquet, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("Written %d rows → %s", df.height, path)


def read_parquet(path: str) -> pl.DataFrame:
    """Read a Parquet file."""
    return pl.read_parquet(path)


def write_csv(df: pl.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)
    logger.info("Written %d rows → %s", df.height, path)


def write_json(data: dict, path: str) -> None:
    """Write a dict as JSON, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Written JSON → %s", path)
