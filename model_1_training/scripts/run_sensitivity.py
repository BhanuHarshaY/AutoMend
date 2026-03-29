"""
run_sensitivity.py

CLI entry point for Captum Integrated Gradients sensitivity analysis.

Usage:
    python -m model_1_training.scripts.run_sensitivity \
        --checkpoint outputs/checkpoints/best_model \
        --split data/splits/val.parquet \
        --output-dir outputs/reports/sensitivity
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

M1_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = M1_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from loguru import logger

from model_1_training.src.utils.config import load_yaml
from model_1_training.src.data.load_parquet import load_track_a, extract_lists
from model_1_training.src.data.tokenizer_setup import build_token_vocab
from model_1_training.src.data.dataset import TrackADataset
from model_1_training.src.eval.evaluator import load_checkpoint
from model_1_training.src.sensitivity.captum_analysis import (
    run_integrated_gradients,
    save_sensitivity_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sensitivity analysis (Captum IG)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-config", default=str(M1_ROOT / "configs/eval/eval.yaml"))
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--anomaly-only", action="store_true",
                        help="Only analyze samples predicted as anomaly (class > 0)")
    args = parser.parse_args()

    eval_cfg = load_yaml(args.eval_config)
    sens_cfg = eval_cfg.get("sensitivity", {})
    n_samples = args.n_samples or sens_cfg.get("n_samples", 200)
    n_steps = args.n_steps or sens_cfg.get("n_steps", 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_checkpoint(args.checkpoint, device=device)

    int_to_token = build_token_vocab()
    df = load_track_a(args.split)
    seqs, labels, _ = extract_lists(df)
    dataset = TrackADataset(seqs, labels, tokenizer, int_to_token, max_length=512)

    target_classes = list(range(1, 7)) if args.anomaly_only else None

    logger.info(f"Running Integrated Gradients on {n_samples} samples ({n_steps} steps)...")
    results = run_integrated_gradients(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        int_to_token=int_to_token,
        n_samples=n_samples,
        n_steps=n_steps,
        device=device,
        target_classes=target_classes,
    )

    save_sensitivity_report(results, args.output_dir)
    print(f"\nSensitivity analysis complete. Reports -> {args.output_dir}")


if __name__ == "__main__":
    main()
