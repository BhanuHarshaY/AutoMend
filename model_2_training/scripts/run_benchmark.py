"""
run_benchmark.py

Locked evaluation entry point — runs the model against the gold benchmark set.

This script is SEPARATE from run_eval.py (validation) and run_test.py (test set).
It exists exclusively to score a checkpoint against the curated gold benchmark,
and may be run as many times as needed for model comparison — it does NOT use
any split that influences training or model selection.

Usage:
    python scripts/run_benchmark.py \\
        --config configs/eval/json_eval.yaml \\
        --checkpoint outputs/checkpoints/best_model

    # Compare a different checkpoint:
    python scripts/run_benchmark.py \\
        --config configs/eval/json_eval.yaml \\
        --checkpoint outputs/checkpoints/checkpoint-300

    # Point to a custom benchmark file:
    python scripts/run_benchmark.py \\
        --config configs/eval/json_eval.yaml \\
        --checkpoint outputs/checkpoints/best_model \\
        --benchmark data/benchmarks/gold_benchmark.jsonl
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

_DEFAULT_BENCHMARK = _M2_ROOT / "data/benchmarks/gold_benchmark.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a checkpoint against the locked gold benchmark."
    )
    parser.add_argument("--config",     required=True, help="Path to eval config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory to evaluate.")
    parser.add_argument(
        "--benchmark",
        default=str(_DEFAULT_BENCHMARK),
        help=f"Gold benchmark JSONL (default: {_DEFAULT_BENCHMARK}).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for benchmark reports (default: outputs/reports/benchmark/<checkpoint_name>).",
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

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        logger.error(
            f"Gold benchmark not found: {benchmark_path}\n"
            "Build it first with:  python scripts/build_benchmark.py"
        )
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_name = checkpoint_path.name

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _M2_ROOT / "outputs/reports/benchmark" / checkpoint_name
    )

    logger.info("=" * 60)
    logger.info("GOLD BENCHMARK EVALUATION")
    logger.info(f"  checkpoint : {checkpoint_path}")
    logger.info(f"  benchmark  : {benchmark_path}")
    logger.info(f"  output     : {output_dir}")
    logger.info("=" * 60)

    metrics = run_evaluation(
        split_path=benchmark_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        eval_cfg=eval_cfg,
        split_name="benchmark",
    )

    logger.success("Gold benchmark evaluation complete.")
    logger.info(f"  json_parse_rate    : {metrics.get('json_parse_rate', 'N/A')}")
    logger.info(f"  non_empty_rate     : {metrics.get('non_empty_rate', 'N/A')}")
    logger.info(f"  tax_valid_rate     : {metrics.get('tax_valid_rate', 'N/A')}")
    logger.info(f"  tax_truncated_rate : {metrics.get('tax_truncated_rate', 'N/A')}")
    logger.info(f"  tax_empty_rate     : {metrics.get('tax_empty_rate', 'N/A')}")
    logger.info(f"  Reports saved in   : {output_dir}")


if __name__ == "__main__":
    main()
