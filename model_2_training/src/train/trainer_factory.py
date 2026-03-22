"""
trainer_factory.py

Assembles a HuggingFace Trainer from model, tokenizer, datasets, collator,
and training config.

Device-aware behaviour
----------------------
  CUDA   bf16=True (Ampere+) or fp16=True — from get_device_config()
  CPU    bf16=False, fp16=False — full fp32 training
  MPS    This module is NOT called on MPS — mlx_train.py handles training.
         If called from MPS by mistake, a RuntimeError is raised.

Note on Transformers 5.x API changes applied here
  - evaluation_strategy → eval_strategy (parameter renamed)
  - tokenizer=           → processing_class= (parameter renamed)

Does NOT load data, models, or tokenizers.
Does NOT run training.
"""

from __future__ import annotations
from pathlib import Path
from loguru import logger

from transformers import TrainingArguments, Trainer

from model_2_training.src.utils.device import detect_device, get_device_config
from model_2_training.src.train.callbacks import WandbStandardCallback


def build_training_args(cfg: dict, output_dir: str | Path) -> TrainingArguments:
    """
    Build TrainingArguments from a config dict, with device-appropriate precision flags.

    The bf16 and fp16 keys in the YAML config are used as a starting point but
    are overridden by what the detected device actually supports:
      - CUDA + Ampere+  → bf16=True,  fp16=False
      - CUDA + older    → bf16=False, fp16=True
      - CPU             → bf16=False, fp16=False

    Args:
        cfg:        Dict from qlora_sft.yaml (train config).
        output_dir: Directory for checkpoints and logs.

    Returns:
        Configured TrainingArguments.

    Raises:
        RuntimeError: if called on MPS.
    """
    device = detect_device()

    if device == "mps":
        raise RuntimeError(
            "build_training_args() was called on MPS. "
            "Apple Silicon training runs via mlx_train.py — check train_loop.py."
        )

    dev_cfg = get_device_config(device, quantization=None)
    bf16 = dev_cfg["bf16"]
    fp16 = dev_cfg["fp16"]

    if bf16:
        logger.info("Precision: bf16 (device supports bfloat16)")
    elif fp16:
        logger.info("Precision: fp16 (bf16 not supported on this GPU)")
    else:
        logger.info("Precision: fp32 (CPU)")

    output_dir = str(output_dir)
    logger.info(f"Building TrainingArguments — output_dir={output_dir}")

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=cfg.get("weight_decay", 0.01),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        # Transformers 5.x: evaluation_strategy renamed to eval_strategy
        eval_strategy=cfg.get("evaluation_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 100),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg.get("greater_is_better", False),
        logging_steps=cfg.get("logging_steps", 10),
        report_to=cfg.get("report_to", "none"),
        # Precision — device-resolved, not blindly taken from config
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
        remove_unused_columns=cfg.get("remove_unused_columns", False),
        ddp_find_unused_parameters=False,
    )


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    data_collator,
    training_args: TrainingArguments,
) -> Trainer:
    """
    Assemble a HuggingFace Trainer with all components.

    Args:
        model:          PEFT model with LoRA adapters attached.
        tokenizer:      Configured tokenizer.
        train_dataset:  ChatMLSupervisedDataset for training.
        eval_dataset:   ChatMLSupervisedDataset for validation.
        data_collator:  AssistantOnlyCollator instance.
        training_args:  TrainingArguments from build_training_args().

    Returns:
        Configured Trainer ready for .train().
    """
    logger.info(
        f"Assembling Trainer — "
        f"train={len(train_dataset)} samples, eval={len(eval_dataset)} samples"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # Transformers 5.x: tokenizer= renamed to processing_class=
        processing_class=tokenizer,
        callbacks=[WandbStandardCallback()],
    )

    logger.info("Trainer assembled successfully.")
    return trainer
