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
import json
import sys
from pathlib import Path

import yaml
from loguru import logger

_M2_ROOT = Path(__file__).resolve().parent.parent
_AUTOMEND_ROOT = _M2_ROOT.parent
sys.path.insert(0, str(_AUTOMEND_ROOT))

from dotenv import load_dotenv
load_dotenv(_AUTOMEND_ROOT / ".env")

from model_2_training.src.eval.evaluator import run_evaluation
from model_2_training.src.tracking.wandb_logger import init_run, log_eval_metrics, finish_run


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
    if not checkpoint_path.is_absolute():
        checkpoint_path = (_M2_ROOT / checkpoint_path).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else _M2_ROOT / "outputs/reports/val"

    # Read the run_name written by training so this eval run clusters with its
    # training run under the same W&B group.
    snapshot_path = checkpoint_path.parent / "run_config_snapshot.json"
    group = None
    if snapshot_path.exists():
        with open(snapshot_path) as f:
            group = json.load(f).get("run_name")

    init_run(
        project="automend-model2",
        run_name=f"eval-val-{group or checkpoint_path.name}",
        tags=["eval", "validation", "phase2"],
        group=group,
    )

    metrics = run_evaluation(
        split_path=split_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        eval_cfg=eval_cfg,
        split_name="validation",
    )

    log_eval_metrics(metrics, split="val")
    finish_run()

    logger.success("Validation evaluation complete.")
    logger.info(f"  json_parse_rate       : {metrics.get('phase1_structural/json_parse_rate', 'N/A')}")
    logger.info(f"  schema_valid_rate     : {metrics.get('phase2a_schema/schema_valid_rate', 'N/A')}")
    logger.info(f"  steps_count_match_rate: {metrics.get('phase2b_fields/steps_count_match_rate', 'N/A')}")
    logger.info(f"  full_param_validity   : {metrics.get('phase2c_params/full_param_validity_rate', 'N/A')}")


if __name__ == "__main__":
    main()
