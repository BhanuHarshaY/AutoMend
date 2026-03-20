"""
run_train.py

Entry point for Model 2 QLoRA supervised fine-tuning.

Usage:
    python scripts/run_train.py \\
        --data-config configs/data/track_b_chatml.yaml \\
        --model-config configs/model/qwen_baseline.yaml \\
        --train-config configs/train/qlora_sft.yaml

All settings are driven by config files. CLI flags allow overrides for
quick experiments without editing YAML files.
"""

from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

# Suppress known harmless deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_reentrant.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.utils.checkpoint.*")

import yaml
from loguru import logger

# Ensure model_2_training parent (AutoMend root) is on sys.path
_M2_ROOT = Path(__file__).resolve().parent.parent
_AUTOMEND_ROOT = _M2_ROOT.parent
sys.path.insert(0, str(_AUTOMEND_ROOT))

from model_2_training.src.train.train_loop import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Model 2 QLoRA supervised fine-tuning."
    )
    parser.add_argument(
        "--data-config",
        required=True,
        help="Path to data config YAML (e.g. configs/data/track_b_chatml.yaml).",
    )
    parser.add_argument(
        "--model-config",
        required=True,
        help="Path to model config YAML (e.g. configs/model/qwen_baseline.yaml).",
    )
    parser.add_argument(
        "--train-config",
        required=True,
        help="Path to training config YAML (e.g. configs/train/qlora_sft.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override num_train_epochs from train config.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning_rate from train config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per_device_train_batch_size from train config.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        logger.error(f"Config not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()

    data_cfg = load_yaml(_M2_ROOT / args.data_config)
    model_cfg = load_yaml(_M2_ROOT / args.model_config)
    train_cfg = load_yaml(_M2_ROOT / args.train_config)

    # Apply CLI overrides
    if args.epochs is not None:
        train_cfg["num_train_epochs"] = args.epochs
        logger.info(f"Override: num_train_epochs={args.epochs}")
    if args.lr is not None:
        train_cfg["learning_rate"] = args.lr
        logger.info(f"Override: learning_rate={args.lr}")
    if args.batch_size is not None:
        train_cfg["per_device_train_batch_size"] = args.batch_size
        logger.info(f"Override: per_device_train_batch_size={args.batch_size}")

    logger.info("=" * 60)
    logger.info("Model 2 Training — QLoRA SFT")
    logger.info(f"  Model  : {model_cfg.get('model_name')}")
    logger.info(f"  Epochs : {train_cfg.get('num_train_epochs')}")
    logger.info(f"  LR     : {train_cfg.get('learning_rate')}")
    logger.info(f"  Quant  : {model_cfg.get('quantization')}")
    logger.info("=" * 60)

    best_model_dir = run_training(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        m2_root=_M2_ROOT,
    )

    logger.success(f"Training complete. Best model saved to: {best_model_dir}")


if __name__ == "__main__":
    main()
