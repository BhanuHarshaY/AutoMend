"""
train_loop.py

Orchestrates the full Model 1 training pipeline:
  1. Set seed
  2. Setup tokenizer with custom MLOps tokens
  3. Load and split data
  4. Build PyTorch Datasets
  5. Load model + resize embeddings
  6. Configure Focal Loss
  7. Build HF Trainer and run training
  8. Save best checkpoint
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score, recall_score
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
)

from model_1_training.src.utils.seed import set_seed
from model_1_training.src.utils.device import detect_device
from model_1_training.src.data.tokenizer_setup import setup_tokenizer
from model_1_training.src.data.load_parquet import load_track_a, get_label_distribution, extract_lists
from model_1_training.src.data.dataset import TrackADataset
from model_1_training.src.data.split_data import stratified_split, save_splits
from model_1_training.src.model.load_model import load_model
from model_1_training.src.model.focal_loss import FocalLoss, compute_class_weights
from model_1_training.src.train.callbacks import LogEpochMetricsCallback, EarlyStoppingWithLogging


class FocalLossTrainer(Trainer):
    """HF Trainer subclass that uses Focal Loss instead of default cross-entropy."""

    def __init__(self, focal_loss_fn: FocalLoss, **kwargs):
        super().__init__(**kwargs)
        self.focal_loss_fn = focal_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _build_compute_metrics(num_labels: int):
    """Return a compute_metrics function for the HF Trainer."""
    label_names_short = list(range(num_labels))

    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        per_class_recall = recall_score(labels, preds, average=None, zero_division=0, labels=label_names_short)
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0, labels=label_names_short)

        anomaly_mask = labels > 0
        anomaly_recall = float(recall_score(
            labels[anomaly_mask], preds[anomaly_mask],
            average="macro", zero_division=0,
            labels=list(range(1, num_labels)),
        )) if anomaly_mask.sum() > 0 else 0.0

        metrics = {
            "macro_f1": macro_f1,
            "anomaly_recall": anomaly_recall,
        }
        for i in range(num_labels):
            metrics[f"recall_class_{i}"] = float(per_class_recall[i]) if i < len(per_class_recall) else 0.0
            metrics[f"f1_class_{i}"] = float(per_class_f1[i]) if i < len(per_class_f1) else 0.0

        return metrics

    return compute_metrics


def save_config_snapshot(
    data_cfg: dict, model_cfg: dict, train_cfg: dict,
    output_dir: Path, device: str,
) -> None:
    """Persist all configs for reproducibility."""
    snapshot = {
        "device": device,
        "data_config": data_cfg,
        "model_config": model_cfg,
        "train_config": train_cfg,
        "timestamp": datetime.now().isoformat(),
    }
    out = output_dir / "run_config_snapshot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(snapshot, f, indent=2)
    logger.info(f"Config snapshot -> {out}")


def run_training(
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    m1_root: Path,
    artifact_path: str | Path | None = None,
    splits_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Full Model 1 training orchestration.

    Args:
        data_cfg: Loaded data config dict.
        model_cfg: Loaded model config dict.
        train_cfg: Loaded training config dict.
        m1_root: Path to model_1_training/ root.
        artifact_path: Override for the combined parquet path.
        splits_dir: Override for the splits directory.
        output_dir: Override for checkpoints output directory.

    Returns:
        Path to the best model checkpoint.
    """
    device = detect_device()
    seed = data_cfg.get("shuffle_seed", 42)
    set_seed(seed)

    num_labels = model_cfg.get("num_labels", data_cfg.get("num_labels", 7))
    max_seq_length = data_cfg.get("max_seq_length", 512)

    _splits_dir = Path(splits_dir) if splits_dir else m1_root / data_cfg.get("splits_dir", "data/splits")
    _output_dir = Path(output_dir) if output_dir else m1_root / train_cfg.get("output_dir", "outputs/checkpoints")
    _output_dir.mkdir(parents=True, exist_ok=True)

    save_config_snapshot(data_cfg, model_cfg, train_cfg, _output_dir, device)

    # --- W&B run name ---
    dt = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"roberta-base_focal_{dt}"
    os.environ["WANDB_RUN_NAME"] = run_name
    os.environ.setdefault("WANDB_PROJECT", "automend-model1")

    # --- Tokenizer ---
    logger.info("Setting up tokenizer with custom MLOps tokens...")
    tokenizer, int_to_token, _ = setup_tokenizer(model_cfg.get("model_name", "roberta-base"))

    # --- Data ---
    train_path = _splits_dir / "train.parquet"
    val_path = _splits_dir / "val.parquet"

    if not train_path.exists():
        logger.info("Splits not found — creating from artifact...")
        _artifact = Path(artifact_path) if artifact_path else m1_root.parent / data_cfg["artifact_path"]
        df = load_track_a(_artifact)
        train_df, val_df, test_df = stratified_split(
            df,
            train_ratio=data_cfg.get("train_ratio", 0.8),
            val_ratio=data_cfg.get("val_ratio", 0.1),
            test_ratio=data_cfg.get("test_ratio", 0.1),
            seed=seed,
        )
        save_splits(train_df, val_df, test_df, _splits_dir)
    else:
        logger.info(f"Loading existing splits from {_splits_dir}")

    import polars as pl
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)

    train_seqs, train_labels, _ = extract_lists(train_df)
    val_seqs, val_labels, _ = extract_lists(val_df)

    logger.info(f"Train: {len(train_labels)} samples, Val: {len(val_labels)} samples")

    train_dataset = TrackADataset(train_seqs, train_labels, tokenizer, int_to_token, max_seq_length)
    val_dataset = TrackADataset(val_seqs, val_labels, tokenizer, int_to_token, max_seq_length)

    # --- Model ---
    model = load_model(
        model_name=model_cfg.get("model_name", "roberta-base"),
        num_labels=num_labels,
        tokenizer=tokenizer,
    )

    # --- Focal Loss ---
    label_dist = get_label_distribution(train_df)
    class_weights = compute_class_weights(label_dist, num_classes=num_labels)
    if device == "cuda":
        class_weights = class_weights.cuda()
    focal_loss_fn = FocalLoss(
        gamma=train_cfg.get("focal_gamma", 2.0),
        alpha=class_weights,
    )

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=str(_output_dir),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 32),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 64),
        num_train_epochs=train_cfg.get("num_train_epochs", 10),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        eval_strategy=train_cfg.get("eval_strategy", "epoch"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=train_cfg.get("save_total_limit", 3),
        logging_steps=train_cfg.get("logging_steps", 50),
        report_to=train_cfg.get("report_to", "wandb"),
        seed=seed,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        fp16=device == "cuda",
    )

    # --- Callbacks ---
    callbacks = [
        LogEpochMetricsCallback(),
        EarlyStoppingWithLogging(
            patience=train_cfg.get("early_stopping_patience", 3),
            metric="eval_macro_f1",
        ),
    ]

    # --- Trainer ---
    trainer = FocalLossTrainer(
        focal_loss_fn=focal_loss_fn,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_build_compute_metrics(num_labels),
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # --- Train ---
    logger.info("=" * 60)
    logger.info("Starting Model 1 training (RoBERTa + Focal Loss)...")
    logger.info("=" * 60)
    trainer.train()

    # --- Save best model ---
    best_model_dir = _output_dir / "best_model"
    logger.info(f"Saving best model -> {best_model_dir}")
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    logger.success(f"Training complete. Best model -> {best_model_dir}")
    return best_model_dir
