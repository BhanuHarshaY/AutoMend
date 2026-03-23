"""
evaluator.py

Orchestrates the full evaluation pipeline for a given split.

Responsibilities:
  - Load samples from a split file
  - Load model and tokenizer from checkpoint
  - Run generation
  - Compute JSON metrics
  - Save all report artifacts
  - Return metrics dict

Shared between validation (run_eval.py) and final testing (run_test.py).
"""

from __future__ import annotations
from pathlib import Path
from loguru import logger

from model_2_training.src.data.load_jsonl import load_jsonl
from model_2_training.src.eval.generator import (
    load_model_for_inference,
    run_generation,
    save_predictions,
)
from model_2_training.src.eval.metrics_json import summarize_errors
from model_2_training.src.eval.metrics_aggregator import (
    run_all_metrics,
    get_per_field_report,
    get_schema_errors,
    get_param_errors,
)
from model_2_training.src.eval.save_reports import save_all_reports


def run_evaluation(
    split_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    eval_cfg: dict,
    split_name: str = "validation",
) -> dict:
    """
    Full evaluation pipeline for one split.

    Args:
        split_path: Path to val.jsonl or test.jsonl.
        checkpoint_path: Path to model checkpoint directory.
        output_dir: Directory to write reports and predictions.
        eval_cfg: Loaded eval config dict (json_eval.yaml).
        split_name: "validation" or "test" — used in report titles.

    Returns:
        Metrics dict.
    """
    split_path = Path(split_path)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    logger.info("=" * 60)
    logger.info(f"Evaluation pipeline — split={split_name}")
    logger.info(f"  checkpoint : {checkpoint_path}")
    logger.info(f"  split      : {split_path}")
    logger.info(f"  output_dir : {output_dir}")
    logger.info("=" * 60)

    # --- Load split data ---
    samples = load_jsonl(split_path, malformed_strategy="skip")
    if not samples:
        raise ValueError(f"Split file is empty or all rows are invalid: {split_path}")
    logger.info(f"Loaded {len(samples)} samples for evaluation")

    # --- Load model ---
    model, tokenizer = load_model_for_inference(checkpoint_path)

    # --- Generate ---
    predictions = run_generation(
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=eval_cfg.get("max_new_tokens", 512),
        temperature=eval_cfg.get("temperature", 0.1),
        top_p=eval_cfg.get("top_p", 0.9),
        do_sample=eval_cfg.get("do_sample", False),
    )

    # --- Save raw predictions ---
    preds_path = output_dir / f"{split_name}_predictions.jsonl"
    save_predictions(predictions, preds_path)

    # --- Compute metrics (all phases) ---
    metrics = run_all_metrics(predictions)

    # --- Summarize errors ---
    n_samples = eval_cfg.get("num_samples_to_save", 50)
    error_samples = summarize_errors(predictions, n_samples=n_samples)

    # --- Per-phase supplementary reports ---
    per_field_report = get_per_field_report(predictions)
    param_errors     = get_param_errors(predictions, n_samples=n_samples)

    # --- Save reports ---
    report_paths = save_all_reports(
        metrics=metrics,
        predictions=predictions,
        error_samples=error_samples,
        output_dir=output_dir,
        split_name=split_name,
        num_sample_outputs=n_samples,
        per_field_report=per_field_report,
        param_errors=param_errors,
    )

    logger.success(f"Evaluation complete. Reports written to: {output_dir}")
    for name, path in report_paths.items():
        logger.info(f"  {name:<20} {path}")

    return metrics
