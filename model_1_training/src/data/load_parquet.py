"""
load_parquet.py

Load Format A parquet data produced by the Track A combiner pipeline.
Handles both the combined file and individual split files.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
from loguru import logger


LABEL_NAMES: List[str] = [
    "Normal",
    "Resource_Exhaustion",
    "System_Crash",
    "Network_Failure",
    "Data_Drift",
    "Auth_Failure",
    "Permission_Denied",
]


def load_track_a(
    path: str | Path,
    label_col: str = "label",
    seq_col: str = "sequence_ids",
) -> pl.DataFrame:
    """
    Load a Format A parquet file.

    Args:
        path: Path to the parquet file.
        label_col: Name of the label column.
        seq_col: Name of the sequence IDs column.

    Returns:
        Polars DataFrame with at least `sequence_ids` and `label` columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Track A data not found: {path}")

    df = pl.read_parquet(path)
    logger.info(f"Loaded {df.height} rows from {path}")
    logger.info(f"Columns: {df.columns}")
    logger.info(f"Schema: {df.schema}")

    for col in (seq_col, label_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")

    label_dist = df[label_col].value_counts().sort(label_col)
    for row in label_dist.iter_rows():
        lbl, cnt = row[0], row[1]
        name = LABEL_NAMES[lbl] if lbl < len(LABEL_NAMES) else f"Unknown_{lbl}"
        logger.info(f"  label {lbl} ({name}): {cnt} rows ({cnt / df.height * 100:.1f}%)")

    return df


def get_label_distribution(df: pl.DataFrame, label_col: str = "label") -> Dict[int, int]:
    """Return {label_int: count} from a DataFrame."""
    dist = df[label_col].value_counts().sort(label_col)
    return dict(zip(dist[label_col].to_list(), dist["count"].to_list()))


def extract_lists(
    df: pl.DataFrame,
    seq_col: str = "sequence_ids",
    label_col: str = "label",
    source_col: Optional[str] = "source_dataset",
) -> Tuple[List[List[int]], List[int], Optional[List[str]]]:
    """
    Extract raw Python lists from a Polars DataFrame for PyTorch consumption.

    Returns:
        (sequence_ids_list, labels_list, sources_list_or_None)
    """
    seqs = df[seq_col].to_list()
    labels = df[label_col].to_list()
    sources = df[source_col].to_list() if source_col and source_col in df.columns else None
    return seqs, labels, sources
