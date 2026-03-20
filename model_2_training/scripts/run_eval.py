"""
run_eval.py

Entry point for running evaluation on the validation set.

Usage:
    python scripts/run_eval.py \\
        --config configs/eval/json_eval.yaml \\
        --checkpoint outputs/checkpoints/best_model

    # Override split path:
    python scripts/run_eval.py \\
        --config configs/eval/json_eval.yaml \\
        --checkpoint outputs/checkpoints/best_model \\
        --split data/splits/val.jsonl
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

from model_2_training.src.eval.evaluator import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Model 2 on the validation set.")
    parser.add_argument("--config", required=True, help="Path to eval config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint directory.")
    parser.add_argument(
        "--split",
        default=None,
        help="Override val split path (default: data/splits/val.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for reports (default: outputs/reports/val).",
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

    split_path = Path(args.split) if args.split else _M2_ROOT / "data/splits/val.jsonl"
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else _M2_ROOT / "outputs/reports/val"

    metrics = run_evaluation(
        split_path=split_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        eval_cfg=eval_cfg,
        split_name="validation",
    )

    logger.success("Validation evaluation complete.")
    logger.info(f"  JSON parse rate : {metrics.get('json_parse_rate', 'N/A')}")
    logger.info(f"  Non-empty rate  : {metrics.get('non_empty_rate', 'N/A')}")


if __name__ == "__main__":
    main()
