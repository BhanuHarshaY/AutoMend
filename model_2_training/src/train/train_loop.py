"""
train_loop.py

Orchestrates the full supervised fine-tuning pipeline for Model 2.

Device dispatch
---------------
  CUDA / CPU  → HuggingFace Transformers + PEFT + Trainer  (HF backend)
  MPS         → mlx-lm LoRA training on Apple Silicon Metal (MLX backend)

The device is detected once at startup. All downstream modules (model loading,
trainer factory, etc.) read the same device rather than detecting independently.

HF backend responsibilities (CUDA / CPU)
  - Set random seed
  - Load tokenizer
  - Load and validate train/val split data
  - Build ChatMLSupervisedDataset objects
  - Build AssistantOnlyCollator
  - Load Qwen with bitsandbytes quantization (CUDA) or fp32 (CPU)
  - Attach LoRA adapters via PEFT
  - Build HuggingFace Trainer and run training
  - Save final adapter + tokenizer

MLX backend responsibilities (MPS)
  - Set random seed
  - Convert JSONL splits to MLX chat format
  - Write mlx-lm YAML config
  - Run `python -m mlx_lm.lora --config ...` via subprocess
  - Save MLX adapter to outputs/checkpoints/best_model/
"""

from __future__ import annotations
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from model_2_training.src.utils.device import detect_device


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def save_config_snapshot(
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    output_dir: Path,
    device: str,
) -> None:
    """Save a snapshot of all configs + detected device for reproducibility."""
    snapshot = {
        "device":       device,
        "data_config":  data_cfg,
        "model_config": model_cfg,
        "train_config": train_cfg,
    }
    snapshot_path = output_dir / "run_config_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    logger.info(f"Config snapshot saved → {snapshot_path}")


# ---------------------------------------------------------------------------
# HuggingFace backend  (CUDA / CPU)
# ---------------------------------------------------------------------------

def _run_hf_training(
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    splits_dir: Path,
    output_dir: Path,
) -> Path:
    """
    Full HF Transformers + PEFT training pipeline for CUDA and CPU.

    Returns:
        Path to best_model/ checkpoint directory.
    """
    from model_2_training.src.model.load_tokenizer import load_tokenizer
    from model_2_training.src.model.load_qwen import load_qwen
    from model_2_training.src.model.lora_factory import build_and_attach_lora
    from model_2_training.src.data.load_jsonl import load_jsonl
    from model_2_training.src.data.dataset_builder import build_dataset
    from model_2_training.src.data.collators import AssistantOnlyCollator
    from model_2_training.src.train.trainer_factory import build_training_args, build_trainer

    max_seq_length      = data_cfg.get("max_seq_length", 2048)
    malformed_strategy  = data_cfg.get("malformed_row_strategy", "skip")

    # --- Tokenizer ---
    tokenizer = load_tokenizer(
        model_cfg["tokenizer_name"],
        trust_remote_code=True,
    )

    # --- Data ---
    train_path = splits_dir / "train.jsonl"
    val_path   = splits_dir / "val.jsonl"

    logger.info(f"Loading train split: {train_path}")
    train_samples = load_jsonl(train_path, malformed_strategy=malformed_strategy)
    logger.info(f"Loading val split: {val_path}")
    val_samples = load_jsonl(val_path, malformed_strategy=malformed_strategy)

    if not train_samples:
        raise RuntimeError(f"Train split is empty: {train_path}. Run run_split.py first.")
    if not val_samples:
        raise RuntimeError(f"Val split is empty: {val_path}. Run run_split.py first.")

    # --- Datasets ---
    logger.info("Building ChatMLSupervisedDataset objects...")
    train_dataset = build_dataset(train_samples, tokenizer, max_seq_length)
    val_dataset   = build_dataset(val_samples,   tokenizer, max_seq_length)

    # --- Collator ---
    collator = AssistantOnlyCollator(tokenizer=tokenizer, max_seq_length=max_seq_length)

    # --- Model ---
    model = load_qwen(
        model_name=model_cfg["model_name"],
        quantization=model_cfg.get("quantization"),
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=True,
    )

    # --- LoRA ---
    is_quantized = model_cfg.get("quantization") is not None
    lora_cfg = {**train_cfg, "lora_target_modules": model_cfg.get("lora_target_modules")}
    model = build_and_attach_lora(model, lora_cfg, is_quantized=is_quantized)

    # --- Trainer ---
    training_args = build_training_args(train_cfg, output_dir=output_dir)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        training_args=training_args,
    )

    # --- Train ---
    logger.info("=" * 60)
    logger.info("Starting HF training (CUDA/CPU backend)...")
    logger.info("=" * 60)
    trainer.train()

    # --- Save ---
    best_model_dir = output_dir / "best_model"
    logger.info(f"Saving best model → {best_model_dir}")
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    return best_model_dir


# ---------------------------------------------------------------------------
# MLX backend  (MPS / Apple Silicon)
# ---------------------------------------------------------------------------

def _run_mlx_training(
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    splits_dir: Path,
    output_dir: Path,
) -> Path:
    """
    MLX + mlx-lm LoRA training pipeline for Apple Silicon (MPS).

    Returns:
        Path to best_model/ adapter directory.
    """
    from model_2_training.src.train.mlx_train import run_mlx_training

    return run_mlx_training(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        splits_dir=splits_dir,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_training(
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    m2_root: Path,
) -> Path:
    """
    Full training orchestration — detects device and routes to the right backend.

    Args:
        data_cfg:  Loaded data config dict (track_b_chatml.yaml).
        model_cfg: Loaded model config dict (qwen_baseline.yaml).
        train_cfg: Loaded train config dict (qlora_sft.yaml).
        m2_root:   Path to model_2_training/ root directory.

    Returns:
        Path to the best checkpoint / adapter directory.
    """
    # --- Device detection (single point) ---
    device = detect_device()

    # --- Seed ---
    seed = data_cfg.get("shuffle_seed", 42)
    set_seed(seed)

    # --- Paths ---
    splits_dir = m2_root / data_cfg.get("splits_dir", "data/splits")
    output_dir = m2_root / train_cfg.get("output_dir", "outputs/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- W&B run name ---
    model_short = model_cfg.get("model_name", "model").split("/")[-1]
    quant       = model_cfg.get("quantization") or ("mlx" if device == "mps" else "fp32")
    lora_r      = train_cfg.get("lora_r", 16)
    n_train     = data_cfg.get("max_train_samples") or "full"
    dt          = datetime.now().strftime("%Y%m%d-%H%M")
    run_name    = f"{model_short}_{quant}_lora-r{lora_r}_{n_train}samples_{dt}"
    os.environ["WANDB_RUN_NAME"] = run_name
    logger.info(f"W&B run name: {run_name}")

    # --- Config snapshot (includes detected device) ---
    save_config_snapshot(data_cfg, model_cfg, train_cfg, output_dir, device)

    # --- Dispatch ---
    logger.info(f"Training backend: {'MLX (Apple Silicon)' if device == 'mps' else 'HuggingFace Transformers'}")

    if device == "mps":
        best_model_dir = _run_mlx_training(
            data_cfg, model_cfg, train_cfg, splits_dir, output_dir
        )
    else:
        best_model_dir = _run_hf_training(
            data_cfg, model_cfg, train_cfg, splits_dir, output_dir
        )

    logger.success(f"Training complete. Best model/adapter → {best_model_dir}")
    return best_model_dir
