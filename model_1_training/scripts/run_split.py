"""
run_split.py

CLI entry point for splitting the Track A combined parquet into
train / val / test splits.

Usage:
    python -m model_1_training.scripts.run_split \
        --config configs/data/track_a.yaml \
        [--artifact /path/to/track_A_combined.parquet] \
        [--splits-dir /path/to/splits]
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

M1_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = M1_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from model_1_training.src.utils.config import load_yaml
from model_1_training.src.data.load_parquet import load_track_a
from model_1_training.src.data.split_data import stratified_split, save_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Track A data")
    parser.add_argument("--config", default=str(M1_ROOT / "configs/data/track_a.yaml"))
    parser.add_argument("--artifact", default=None, help="Override artifact_path")
    parser.add_argument("--splits-dir", default=None, help="Override splits output dir")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    artifact_path = Path(args.artifact) if args.artifact else PROJECT_ROOT / cfg["artifact_path"]
    splits_dir = Path(args.splits_dir) if args.splits_dir else M1_ROOT / cfg.get("splits_dir", "data/splits")

    logger.info(f"Artifact: {artifact_path}")
    logger.info(f"Splits dir: {splits_dir}")

    df = load_track_a(artifact_path)
    train_df, val_df, test_df = stratified_split(
        df,
        train_ratio=cfg.get("train_ratio", 0.8),
        val_ratio=cfg.get("val_ratio", 0.1),
        test_ratio=cfg.get("test_ratio", 0.1),
        seed=cfg.get("shuffle_seed", 42),
    )
    save_splits(train_df, val_df, test_df, splits_dir)
    logger.success("Splits saved successfully")


if __name__ == "__main__":
    main()
