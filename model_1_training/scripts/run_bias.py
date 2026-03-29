"""
run_bias.py

CLI entry point for Fairlearn bias detection on Model 1 predictions.

Usage:
    python -m model_1_training.scripts.run_bias \
        --checkpoint outputs/checkpoints/best_model \
        --split data/splits/val.parquet \
        --output-dir outputs/reports/bias
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

M1_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = M1_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from loguru import logger

from model_1_training.src.utils.config import load_yaml
from model_1_training.src.data.load_parquet import load_track_a, extract_lists
from model_1_training.src.data.tokenizer_setup import build_token_vocab
from model_1_training.src.data.dataset import TrackADataset
from model_1_training.src.eval.evaluator import load_checkpoint, run_inference
from model_1_training.src.bias.fairlearn_check import (
    run_bias_detection,
    compute_sampler_weights,
    save_bias_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bias detection (Fairlearn)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-config", default=str(M1_ROOT / "configs/eval/eval.yaml"))
    parser.add_argument("--disparity-threshold", type=float, default=None)
    args = parser.parse_args()

    eval_cfg = load_yaml(args.eval_config)
    bias_cfg = eval_cfg.get("bias", {})
    threshold = args.disparity_threshold or bias_cfg.get("disparity_threshold", 0.10)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_checkpoint(args.checkpoint, device=device)

    int_to_token = build_token_vocab()
    df = load_track_a(args.split)
    seqs, labels, sources = extract_lists(df)

    if sources is None:
        logger.error("No 'source_dataset' column found — cannot perform slice-based bias detection")
        sys.exit(1)

    dataset = TrackADataset(seqs, labels, tokenizer, int_to_token, max_length=512)

    logger.info(f"Running inference on {len(dataset)} samples for bias analysis...")
    y_pred, y_true, _ = run_inference(model, dataset, batch_size=64, device=device)

    sensitive_features = pd.DataFrame({"source_dataset": sources})

    results = run_bias_detection(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        disparity_threshold=threshold,
    )

    if results["bias_detected"]:
        logger.warning("Bias detected! Computing mitigation sampler weights...")
        weights = compute_sampler_weights(sources, y_true)
        results["mitigation"] = {
            "strategy": "WeightedRandomSampler",
            "weight_stats": {
                "min": float(weights.min()),
                "max": float(weights.max()),
                "mean": float(weights.mean()),
            },
        }
        np.save(Path(args.output_dir) / "sampler_weights.npy", weights)
    else:
        logger.success("No significant bias detected across slices")

    save_bias_report(results, args.output_dir)

    print(f"\nBias detection complete.")
    print(f"Bias detected: {results['bias_detected']}")
    print(f"Reports -> {args.output_dir}")


if __name__ == "__main__":
    main()
