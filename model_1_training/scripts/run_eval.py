"""
run_eval.py

CLI entry point for evaluating a trained Model 1 checkpoint.

Usage:
    python -m model_1_training.scripts.run_eval \
        --checkpoint outputs/checkpoints/best_model \
        --split data/splits/val.parquet \
        --output-dir outputs/reports/val
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
from model_1_training.src.eval.evaluator import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Model 1 checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--split", required=True, help="Path to evaluation split parquet")
    parser.add_argument("--output-dir", required=True, help="Directory for reports/plots")
    parser.add_argument("--eval-config", default=str(M1_ROOT / "configs/eval/eval.yaml"))
    parser.add_argument("--num-labels", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    eval_cfg = load_yaml(args.eval_config)
    batch_size = args.batch_size or eval_cfg.get("per_device_eval_batch_size", 64)

    metrics = evaluate(
        checkpoint_dir=args.checkpoint,
        split_path=args.split,
        output_dir=args.output_dir,
        num_labels=args.num_labels,
        batch_size=batch_size,
    )

    threshold = 0.90
    if metrics["macro_f1"] >= threshold:
        logger.success(f"PASSED: Macro F1 = {metrics['macro_f1']:.4f} >= {threshold}")
    else:
        logger.warning(f"BELOW THRESHOLD: Macro F1 = {metrics['macro_f1']:.4f} < {threshold}")

    print(f"\nMacro F1: {metrics['macro_f1']:.4f}")
    print(f"Anomaly Recall: {metrics['anomaly_recall']:.4f}")
    print(f"Reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
