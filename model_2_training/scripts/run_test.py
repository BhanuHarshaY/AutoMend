"""
run_test.py

Entry point for final evaluation on the held-out test set.

IMPORTANT: Only run this ONCE on the best model after all training and
validation-based model selection is complete. Do not use test results
for any model selection or hyperparameter decisions.

Usage:
    python scripts/run_test.py \\
        --config configs/eval/json_eval.yaml \\
        --checkpoint outputs/checkpoints/best_model

    # Override test split path:
    python scripts/run_test.py \\
        --config configs/eval/json_eval.yaml \\
        --checkpoint outputs/checkpoints/best_model \\
        --split data/splits/test.jsonl
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger

_M2_ROOT = Path(__file__).resolve().parent.parent
_AUTOMEND_ROOT = _M2_ROOT.parent
sys.path.insert(0, str(_AUTOMEND_ROOT))

from model_2_training.src.test.run_testset import run_test_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FINAL evaluation on the held-out test set. Use only once on best model."
    )
    parser.add_argument("--config", required=True, help="Path to eval config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to best model checkpoint.")
    parser.add_argument(
        "--split",
        default=None,
        help="Override test split path (default: data/splits/test.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: outputs/reports/test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = _M2_ROOT / args.config
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        eval_cfg = yaml.safe_load(f)

    split_path = Path(args.split) if args.split else _M2_ROOT / "data/splits/test.jsonl"
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else _M2_ROOT / "outputs/reports/test"

    logger.warning("=" * 60)
    logger.warning("FINAL TEST EVALUATION — use only once on best model")
    logger.warning(f"  checkpoint : {checkpoint_path}")
    logger.warning(f"  test split : {split_path}")
    logger.warning("=" * 60)

    metrics = run_test_evaluation(
        test_split_path=split_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        eval_cfg=eval_cfg,
    )

    logger.success("Final benchmark complete.")
    logger.info(f"  JSON parse rate : {metrics.get('json_parse_rate', 'N/A')}")
    logger.info(f"  Non-empty rate  : {metrics.get('non_empty_rate', 'N/A')}")
    logger.info(f"  Reports in      : {output_dir}")


if __name__ == "__main__":
    main()
