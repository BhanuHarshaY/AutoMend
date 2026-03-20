"""
run_testset.py

Runs the final evaluation on the held-out test set using the best checkpoint.

IMPORTANT: The test set must NEVER be used during model selection or hyperparameter
tuning. This script is only run once on the final best model.

Usage (via scripts/run_test.py — do not call this directly):
    Called by run_test.py after the best checkpoint is confirmed.
"""

from __future__ import annotations
from pathlib import Path
from loguru import logger

from model_2_training.src.eval.evaluator import run_evaluation


def run_test_evaluation(
    test_split_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    eval_cfg: dict,
) -> dict:
    """
    Run final evaluation on the test set.

    Args:
        test_split_path: Path to test.jsonl.
        checkpoint_path: Path to the best model checkpoint.
        output_dir: Directory to write final benchmark report.
        eval_cfg: Eval config dict.

    Returns:
        Final metrics dict.
    """
    logger.warning(
        "Running FINAL TEST EVALUATION. "
        "This should only be called once on the best model. "
        "Do not use these results for model selection."
    )

    metrics = run_evaluation(
        split_path=test_split_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        eval_cfg=eval_cfg,
        split_name="test",
    )

    logger.success("Final test evaluation complete.")
    return metrics
