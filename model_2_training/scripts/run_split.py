"""
run_split.py

Entry point for creating train/val/test splits from the Track B artifact.

Usage:
    python scripts/run_split.py --config configs/data/track_b_chatml.yaml

    # Override artifact path:
    python scripts/run_split.py --config configs/data/track_b_chatml.yaml \
        --artifact path/to/track_B_combined.jsonl
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Allow running from model_2_training/ root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))  # AutoMend root on path

import yaml
from loguru import logger

from model_2_training.src.data.split_data import run_split


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the split pipeline."""
    parser = argparse.ArgumentParser(description="Create train/val/test splits for Model 2 training.")
    parser.add_argument("--config", required=True, help="Path to data config YAML.")
    parser.add_argument("--artifact", default=None, help="Override artifact_path from config.")
    parser.add_argument("--splits-dir", default=None, help="Override splits_dir from config.")
    parser.add_argument("--seed", type=int, default=None, help="Override shuffle_seed from config.")
    return parser.parse_args()


def main() -> None:
    """Run the full split pipeline from CLI arguments and a YAML config."""
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve paths relative to model_2_training/ root
    m2_root = Path(__file__).resolve().parent.parent

    artifact_path = Path(args.artifact) if args.artifact else m2_root.parent / cfg["artifact_path"]
    splits_dir = Path(args.splits_dir) if args.splits_dir else m2_root / cfg["splits_dir"]
    seed = args.seed if args.seed is not None else cfg.get("shuffle_seed", 42)

    summary = run_split(
        artifact_path=artifact_path,
        splits_dir=splits_dir,
        train_ratio=cfg.get("train_ratio", 0.8),
        val_ratio=cfg.get("val_ratio", 0.1),
        test_ratio=cfg.get("test_ratio", 0.1),
        seed=seed,
        malformed_strategy=cfg.get("malformed_row_strategy", "skip"),
        stratify_by=cfg.get("stratify_by"),
        max_train_samples=cfg.get("max_train_samples"),
        max_val_samples=cfg.get("max_val_samples"),
        max_test_samples=cfg.get("max_test_samples"),
    )

    logger.success("Split pipeline complete.")
    logger.info(f"  train : {summary['train_size']} samples")
    logger.info(f"  val   : {summary['val_size']} samples")
    logger.info(f"  test  : {summary['test_size']} samples")
    logger.info(f"  saved : {summary['splits_dir']}")


if __name__ == "__main__":
    main()
