"""
split_data.py

Stratified train / val / test splitting for Track A data.
Writes each split as a separate parquet file under the splits directory.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import polars as pl
from loguru import logger
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pl.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    label_col: str = "label",
    seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split a DataFrame into train / val / test using stratified sampling.

    Args:
        df: Input DataFrame with a label column.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        label_col: Column to stratify on.
        seed: Random seed.

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # Ensure every class has at least 3 rows (minimum for a 3-way stratified split)
    min_per_class = 3
    label_counts = df[label_col].value_counts()
    rare_rows = []
    for row in label_counts.iter_rows():
        lbl, cnt = row[0], row[1]
        if cnt < min_per_class:
            subset = df.filter(pl.col(label_col) == lbl)
            needed = min_per_class - cnt
            rare_rows.append(subset.sample(n=needed, with_replacement=True, seed=seed))
            logger.warning(
                f"Class {lbl} has only {cnt} sample(s) — duplicated to {min_per_class} "
                f"for stratified splitting"
            )
    if rare_rows:
        df = pl.concat([df] + rare_rows)

    labels = df[label_col].to_list()
    indices = list(range(df.height))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1 - train_ratio),
        stratify=labels,
        random_state=seed,
    )

    temp_labels = [labels[i] for i in temp_idx]
    relative_test = test_ratio / (val_ratio + test_ratio)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=seed,
    )

    train_df = df[train_idx]
    val_df = df[val_idx]
    test_df = df[test_idx]

    logger.info(
        f"Split sizes — train: {train_df.height}, val: {val_df.height}, test: {test_df.height}"
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df[label_col].value_counts().sort(label_col)
        logger.info(f"  {name} label distribution: {dict(zip(dist[label_col].to_list(), dist['count'].to_list()))}")

    return train_df, val_df, test_df


def save_splits(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write splits as parquet files and return their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        p = output_dir / f"{name}.parquet"
        split_df.write_parquet(p)
        paths[name] = p
        logger.info(f"Saved {name} split ({split_df.height} rows) -> {p}")

    summary = {
        "train_rows": train_df.height,
        "val_rows": val_df.height,
        "test_rows": test_df.height,
    }
    import json
    with open(output_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return paths
