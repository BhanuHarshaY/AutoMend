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
from model_2_training.src.tracking.wandb_logger import init_run, log_eval_metrics, finish_run


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
    if not checkpoint_path.is_absolute():
        checkpoint_path = (_M2_ROOT / checkpoint_path).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else _M2_ROOT / "outputs/reports/test"

    logger.warning("=" * 60)
    logger.warning("FINAL TEST EVALUATION — use only once on best model")
    logger.warning(f"  checkpoint : {checkpoint_path}")
    logger.warning(f"  test split : {split_path}")
    logger.warning("=" * 60)

    init_run(
        project="automend-model2",
        run_name=f"eval-test-{checkpoint_path.name}",
        tags=["eval", "test", "phase2", "final"],
    )

    metrics = run_test_evaluation(
        test_split_path=split_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        eval_cfg=eval_cfg,
    )

    log_eval_metrics(metrics, split="test")
    finish_run()

    logger.success("Final benchmark complete.")
    logger.info(f"  json_parse_rate       : {metrics.get('phase1_structural/json_parse_rate', 'N/A')}")
    logger.info(f"  schema_valid_rate     : {metrics.get('phase2a_schema/schema_valid_rate', 'N/A')}")
    logger.info(f"  steps_count_match_rate: {metrics.get('phase2b_fields/steps_count_match_rate', 'N/A')}")
    logger.info(f"  full_param_validity   : {metrics.get('phase2c_params/full_param_validity_rate', 'N/A')}")
    logger.info(f"  Reports in            : {output_dir}")


if __name__ == "__main__":
    main()
