"""
evaluator.py

Run inference on a split and compute all evaluation metrics + plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

from model_1_training.src.data.load_parquet import load_track_a, extract_lists
from model_1_training.src.data.tokenizer_setup import setup_tokenizer
from model_1_training.src.data.dataset import TrackADataset
from model_1_training.src.eval.metrics import compute_all_metrics
from model_1_training.src.eval.plots import plot_confusion_matrix, plot_per_class_metrics


def load_checkpoint(
    checkpoint_dir: str | Path,
    num_labels: int = 7,
    device: str = "cpu",
) -> tuple:
    """Load a saved model + tokenizer checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    logger.info(f"Loading checkpoint from {checkpoint_dir}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(checkpoint_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir), num_labels=num_labels,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def run_inference(
    model: AutoModelForSequenceClassification,
    dataset: TrackADataset,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference and return predictions, true labels, and logits.

    Returns:
        (y_pred, y_true, logits) as numpy arrays.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []
    all_logits = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()
        preds = np.argmax(logits, axis=-1)

        all_preds.append(preds)
        all_labels.append(labels.numpy())
        all_logits.append(logits)

    return np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_logits)


def evaluate(
    checkpoint_dir: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    num_labels: int = 7,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> Dict:
    """
    Full evaluation pipeline: load model, run inference, compute metrics, save plots.

    Args:
        checkpoint_dir: Path to saved model checkpoint.
        split_path: Path to the parquet split to evaluate on.
        output_dir: Directory to save reports and plots.
        num_labels: Number of classes.
        batch_size: Inference batch size.
        device: Device override (auto-detected if None).

    Returns:
        Dict of all computed metrics.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_checkpoint(checkpoint_dir, num_labels, device)

    from model_1_training.src.data.tokenizer_setup import build_token_vocab
    int_to_token = build_token_vocab()

    df = load_track_a(split_path)
    seqs, labels, sources = extract_lists(df)
    dataset = TrackADataset(seqs, labels, tokenizer, int_to_token, max_length=512)

    logger.info(f"Running inference on {len(dataset)} samples...")
    y_pred, y_true, logits = run_inference(model, dataset, batch_size, device)

    metrics = compute_all_metrics(y_true, y_pred, num_labels)

    with open(output_dir / "metrics.json", "w") as f:
        serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
        json.dump(serializable, f, indent=2)

    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(metrics["classification_report"])

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        output_dir / "confusion_matrix.png",
    )

    plot_per_class_metrics(
        metrics["per_class"],
        output_dir / "per_class_f1.png",
        metric="f1",
        title="Per-Class F1 Score",
    )
    plot_per_class_metrics(
        metrics["per_class"],
        output_dir / "per_class_recall.png",
        metric="recall",
        title="Per-Class Recall",
    )

    np.save(output_dir / "predictions.npy", y_pred)
    np.save(output_dir / "logits.npy", logits)

    logger.success(f"Evaluation complete. Reports -> {output_dir}")
    return metrics
