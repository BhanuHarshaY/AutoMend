"""
split_data.py

Creates reproducible train/validation/test splits from the validated JSONL artifact.

Responsibilities:
  - Load validated samples via load_jsonl
  - Shuffle with a fixed seed
  - Split into train/val/test by configured ratios
  - Optionally stratify by metadata fields
  - Save output JSONL files to splits_dir
  - Save a split summary report

Does NOT tokenize or train.
"""

from __future__ import annotations
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from model_2_training.src.data.load_jsonl import load_jsonl


def _save_jsonl(samples: list[dict], path: Path) -> None:
    """Write a list of dicts to a JSONL file, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(samples)} samples → {path}")


def _get_stratify_key(sample: dict, fields: list[str]) -> str:
    """Extract a composite stratification key from a sample's metadata."""
    meta = sample.get("metadata") or sample
    parts = []
    for field in fields:
        val = meta.get(field, "unknown")
        parts.append(str(val))
    return "|".join(parts)


def split_samples(
    samples: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_by: list[str] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split samples into train/val/test sets.

    Args:
        samples: List of validated sample dicts.
        train_ratio: Fraction for training (e.g. 0.8).
        val_ratio: Fraction for validation (e.g. 0.1).
        test_ratio: Fraction for test (e.g. 0.1).
        seed: Random seed for reproducibility.
        stratify_by: Optional list of metadata fields to stratify on.

    Returns:
        (train, val, test) lists.

    Raises:
        ValueError: if ratios don't approximately sum to 1.0.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

    rng = random.Random(seed)

    if stratify_by:
        # Group by stratification key
        groups: dict[str, list[dict]] = defaultdict(list)
        for s in samples:
            key = _get_stratify_key(s, stratify_by)
            groups[key].append(s)

        train, val, test = [], [], []
        for key, group in sorted(groups.items()):
            rng.shuffle(group)
            n = len(group)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            # test gets remainder
            train.extend(group[:n_train])
            val.extend(group[n_train:n_train + n_val])
            test.extend(group[n_train + n_val:])

        logger.info(f"Stratified split across {len(groups)} groups: {list(groups.keys())[:5]}...")
    else:
        shuffled = list(samples)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]

    logger.info(
        f"Split result — train: {len(train)}, val: {len(val)}, test: {len(test)} "
        f"(total: {len(train)+len(val)+len(test)})"
    )
    return train, val, test


def run_split(
    artifact_path: str | Path,
    splits_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    malformed_strategy: str = "skip",
    stratify_by: list[str] | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, Any]:
    """
    Full split pipeline: load -> split -> save -> return summary.

    Args:
        artifact_path: Path to track_B_combined.jsonl.
        splits_dir: Directory to save train/val/test.jsonl.
        train_ratio: Training fraction.
        val_ratio: Validation fraction.
        test_ratio: Test fraction.
        seed: Shuffle seed.
        malformed_strategy: "skip" or "raise".
        stratify_by: Optional metadata fields to stratify on.

    Returns:
        Summary dict with split sizes and paths.
    """
    artifact_path = Path(artifact_path)
    splits_dir = Path(splits_dir)

    logger.info(f"Starting split pipeline: {artifact_path} -> {splits_dir}")

    samples = load_jsonl(artifact_path, malformed_strategy=malformed_strategy)

    if len(samples) == 0:
        raise ValueError("No valid samples found in artifact. Cannot split an empty dataset.")

    train, val, test = split_samples(
        samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify_by=stratify_by,
    )

    # Apply sample caps if configured
    if max_train_samples and len(train) > max_train_samples:
        train = train[:max_train_samples]
        logger.info(f"Train capped at {max_train_samples} samples")
    if max_val_samples and len(val) > max_val_samples:
        val = val[:max_val_samples]
        logger.info(f"Val capped at {max_val_samples} samples")
    if max_test_samples and len(test) > max_test_samples:
        test = test[:max_test_samples]
        logger.info(f"Test capped at {max_test_samples} samples")

    train_path = splits_dir / "train.jsonl"
    val_path = splits_dir / "val.jsonl"
    test_path = splits_dir / "test.jsonl"

    _save_jsonl(train, train_path)
    _save_jsonl(val, val_path)
    _save_jsonl(test, test_path)

    summary = {
        "artifact_path": str(artifact_path),
        "seed": seed,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "total_valid": len(samples),
        "splits_dir": str(splits_dir),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
    }

    # Save summary
    summary_path = splits_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Split summary saved → {summary_path}")

    return summary
