"""
mlx_train.py

MLX backend for LoRA supervised fine-tuning on Apple Silicon (MPS).

This module is only imported and executed when the detected device is "mps".
On CUDA or CPU, the standard HuggingFace + PEFT trainer is used instead.

Why MLX instead of bitsandbytes on MPS?
  bitsandbytes implements quantization as CUDA C++ kernels — they cannot run on
  Apple's Metal GPU. MLX is Apple's own ML framework that provides native Metal
  kernels for training and inference, including 4-bit quantization support.

Flow
----
  1. prepare_mlx_data()     Convert our JSONL splits to MLX chat JSONL format
                            (train.jsonl → train.jsonl, val.jsonl → valid.jsonl)
  2. build_mlx_lora_config()  Write a YAML config for mlx_lm.lora
  3. run_mlx_training()     Call mlx_lm.lora via subprocess, save adapter
  4. run_mlx_inference()    Load adapter via mlx_lm.load, generate via mlx_lm.generate

Data format expected by mlx-lm (chat)
--------------------------------------
  {"messages": [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."}
  ]}

  Our track_B_combined.jsonl already uses this structure — no reformatting needed,
  just copy and rename the files.

Requires
--------
  pip install mlx mlx-lm
  macOS 13.5+ on Apple Silicon (M1 / M2 / M3 / M4)
"""

from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from loguru import logger


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_mlx_data(splits_dir: Path, mlx_data_dir: Path) -> None:
    """
    Copy train/val/test splits to a directory that mlx-lm expects.

    mlx-lm requires:
        <data_dir>/train.jsonl   — training samples
        <data_dir>/valid.jsonl   — validation samples  (note: "valid", not "val")
        <data_dir>/test.jsonl    — test samples (optional)

    Each line must be: {"messages": [...]}
    Our splits already carry this structure; we extract the "messages" field
    and strip any other keys (metadata, etc.) that mlx-lm doesn't need.

    Args:
        splits_dir:   Directory containing train.jsonl, val.jsonl, test.jsonl
        mlx_data_dir: Output directory for MLX-formatted data
    """
    mlx_data_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "train.jsonl": "train.jsonl",
        "val.jsonl":   "valid.jsonl",   # mlx-lm expects "valid" not "val"
        "test.jsonl":  "test.jsonl",
    }

    for src_name, dst_name in file_map.items():
        src = splits_dir / src_name
        dst = mlx_data_dir / dst_name

        if not src.exists():
            logger.warning(f"Split file not found, skipping: {src}")
            continue

        count = 0
        with open(src, encoding="utf-8") as f_in, open(dst, "w", encoding="utf-8") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if "messages" not in sample:
                    logger.warning(f"Sample missing 'messages' key, skipping: {line[:80]}")
                    continue
                # Write only the messages field — mlx-lm chat format
                mlx_row = {"messages": sample["messages"]}
                f_out.write(json.dumps(mlx_row, ensure_ascii=False) + "\n")
                count += 1

        logger.info(f"MLX data: {src_name} → {dst_name} ({count} samples)")


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _compute_iters(train_cfg: dict, train_jsonl: Path) -> int:
    """Compute total training iterations from epochs, samples, and batch size."""
    n_samples = sum(1 for _ in open(train_jsonl, encoding="utf-8"))
    batch_size = max(1, train_cfg.get("per_device_train_batch_size", 2))
    epochs = train_cfg.get("num_train_epochs", 3)
    iters = max(1, int((n_samples / batch_size) * epochs))
    logger.info(
        f"MLX iters = ceil({n_samples} samples / {batch_size} batch_size) "
        f"× {epochs} epochs = {iters}"
    )
    return iters


def build_mlx_lora_config(
    model_cfg: dict,
    train_cfg: dict,
    mlx_data_dir: Path,
    adapter_path: Path,
    iters: int,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> dict:
    """
    Build the YAML config dict for mlx_lm.lora.

    mlx-lm uses a flat YAML config rather than CLI flags when --config is passed.
    This gives us full control over all LoRA and training hyperparameters.

    Key parameter notes:
        lora_layers: Number of transformer layers (counting from the last) to apply
                     LoRA to. Different from lora_r (rank). Default 16 covers most
                     of a 1.5B model (which has 28 layers total).
        lora_parameters.rank: The LoRA rank r — matches our lora_r config key.
        lora_parameters.alpha: LoRA scaling factor — matches lora_alpha.
    """
    return {
        "model":        model_cfg["model_name"],
        "train":        True,
        "data":         str(mlx_data_dir),
        "adapter_path": str(adapter_path),

        # Training schedule
        "iters":              iters,
        "batch_size":         train_cfg.get("per_device_train_batch_size", 2),
        "lr_schedule": {
            "name":      "cosine_decay",
            "arguments": [float(train_cfg.get("learning_rate", 2e-4)), iters],
        },
        "warmup":             max(1, int(iters * train_cfg.get("warmup_ratio", 0.05))),
        "max_seq_length":     2048,

        # LoRA
        "lora_layers": 16,    # transformer layers to adapt (from the end)
        "lora_parameters": {
            "rank":    train_cfg.get("lora_r", 16),
            "alpha":   train_cfg.get("lora_alpha", 32),
            "dropout": train_cfg.get("lora_dropout", 0.05),
            "scale":   1.0,
        },

        # Reporting / checkpointing
        "steps_per_report": train_cfg.get("logging_steps", 10),
        "steps_per_eval":   train_cfg.get("eval_steps", 100),
        "val_batches":      25,
        "save_every":       train_cfg.get("save_steps", 100),
        "grad_checkpoint":  True,   # saves memory on MPS

        # W&B tracking (only added when project is configured)
        # report_to must be set to "wandb" to activate the WandBCallback in mlx-lm;
        # wandb_project alone is not enough — mlx-lm defaults report_to to None.
        **( {"report_to": "wandb", "wandb_project": wandb_project, "wandb_run_name": wandb_run_name}
            if wandb_project else {} ),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_mlx_training(
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    splits_dir: Path,
    output_dir: Path,
) -> Path:
    """
    Run LoRA fine-tuning using mlx-lm on Apple Silicon.

    Steps:
        1. Verify mlx is installed
        2. Prepare data in MLX chat format
        3. Write mlx-lm YAML config
        4. Invoke `python -m mlx_lm.lora --config <config.yaml>`
        5. Return path to the saved adapter directory

    Args:
        data_cfg:    Loaded data config dict
        model_cfg:   Loaded model config dict
        train_cfg:   Loaded training config dict
        splits_dir:  Directory containing train.jsonl / val.jsonl
        output_dir:  Output root (checkpoints directory)

    Returns:
        Path to the saved MLX adapter directory (best_model/).

    Raises:
        ImportError: if mlx or mlx-lm are not installed
        subprocess.CalledProcessError: if mlx_lm.lora exits with an error
    """
    # Verify mlx is available before doing anything else
    try:
        import mlx.core  # noqa
    except ImportError:
        raise ImportError(
            "MLX is not installed. Install it with:\n"
            "    pip install mlx mlx-lm\n"
            "MLX requires macOS 13.5+ on Apple Silicon (M1/M2/M3/M4)."
        )

    mlx_data_dir = output_dir / "mlx_data"
    config_path  = output_dir / "mlx_lora_config.yaml"

    # --- Prepare data ---
    logger.info("Preparing MLX chat-format data...")
    prepare_mlx_data(splits_dir, mlx_data_dir)

    # --- Compute iters ---
    iters = _compute_iters(train_cfg, mlx_data_dir / "train.jsonl")

    # --- Write YAML config ---
    wandb_project  = os.environ.get("WANDB_PROJECT")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME")
    # mlx-lm names the W&B run after os.path.basename(adapter_path), so use the
    # run name as the adapter directory so it appears correctly in the W&B UI.
    # After training we rename it to best_model/ for consistent downstream access.
    named_adapter_path = output_dir / (wandb_run_name or "best_model")
    best_model_path    = output_dir / "best_model"
    mlx_cfg = build_mlx_lora_config(
        model_cfg, train_cfg, mlx_data_dir, named_adapter_path, iters,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    with open(config_path, "w") as f:
        yaml.dump(mlx_cfg, f, default_flow_style=False, sort_keys=False)
    logger.info(f"MLX config written → {config_path}")

    # --- Run training ---
    logger.info("=" * 60)
    logger.info("Starting MLX LoRA training on Apple Silicon...")
    logger.info(f"  Model   : {model_cfg['model_name']}")
    logger.info(f"  Iters   : {iters}")
    logger.info(f"  LoRA r  : {train_cfg.get('lora_r', 16)}")
    logger.info(f"  LR      : {train_cfg.get('learning_rate', 2e-4)}")
    logger.info(f"  Adapter : {named_adapter_path}")
    logger.info("=" * 60)

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)]
    subprocess.run(cmd, check=True)

    # Rename run-named adapter dir → best_model/ for consistent downstream access
    if named_adapter_path != best_model_path and named_adapter_path.exists():
        if best_model_path.exists():
            shutil.rmtree(best_model_path)
        named_adapter_path.rename(best_model_path)
        logger.info(f"Adapter renamed: {named_adapter_path.name}/ → best_model/")

    logger.success(f"MLX training complete. Adapter saved → {best_model_path}")
    return best_model_path


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_mlx_inference(
    samples: list[dict],
    checkpoint_path: Path,
    model_name: str,
    max_new_tokens: int = 512,
) -> list[dict]:
    """
    Run generation using mlx-lm on Apple Silicon.

    Handles two checkpoint types:
        - MLX adapter directory (contains adapter_config.json from mlx-lm)
          → loads base model + adapter via mlx_lm.load(base, adapter_path=...)
        - Full MLX model directory
          → loads directly via mlx_lm.load(path)

    Args:
        samples:          List of validated sample dicts with 'messages'.
        checkpoint_path:  Path to the MLX adapter or model directory.
        model_name:       Base model HF name (fallback if not in adapter config).
        max_new_tokens:   Max tokens to generate per sample.

    Returns:
        List of dicts: {index, prompt, generated, reference, sample}
    """
    try:
        from mlx_lm import load, generate
    except ImportError:
        raise ImportError(
            "mlx-lm is not installed. Install with:\n"
            "    pip install mlx mlx-lm"
        )

    checkpoint_path = Path(checkpoint_path)
    adapter_config_path = checkpoint_path / "adapter_config.json"

    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            ac = json.load(f)
        base_name = ac.get("base_model_name_or_path", model_name)
        logger.info(f"Loading base model '{base_name}' + MLX adapter from {checkpoint_path}")
        model, tokenizer = load(base_name, adapter_path=str(checkpoint_path))
    else:
        logger.info(f"Loading MLX model from {checkpoint_path}")
        model, tokenizer = load(str(checkpoint_path))

    logger.info(f"Running MLX inference on {len(samples)} samples...")

    results = []
    for i, sample in enumerate(samples):
        try:
            messages = sample["messages"]

            # Build prompt: everything up to and including the last user message
            last_user_idx = max(
                j for j, m in enumerate(messages) if m["role"] == "user"
            )
            prompt_messages = messages[: last_user_idx + 1]

            # Reference: last assistant message
            reference = next(
                (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
                "",
            )

            # Apply chat template (Qwen uses its own template)
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            generated = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                verbose=False,
            )

            results.append({
                "index":     i,
                "prompt":    prompt,
                "generated": generated,
                "reference": reference,
                "sample":    sample,
            })

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"MLX generated {i + 1}/{len(samples)}")

        except Exception as e:
            logger.warning(f"Sample {i} MLX generation failed: {e}")
            results.append({
                "index":     i,
                "prompt":    "",
                "generated": "",
                "reference": "",
                "sample":    sample,
                "error":     str(e),
            })

    logger.info(f"MLX inference complete — {len(results)} samples processed")
    return results
