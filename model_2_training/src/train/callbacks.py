"""
callbacks.py

Custom training callbacks for Model 2 SFT.

Provides:
  - LoggingCallback: logs train/eval loss to loguru at each logging step
  - GPUMetricsCallback: logs GPU memory usage and timing to W&B via the
    HuggingFace logs dict (WandbCallback picks these up automatically)
  - EarlyStoppingCallback is handled natively by HuggingFace Trainer via
    load_best_model_at_end + metric_for_best_model in TrainingArguments.
"""

from __future__ import annotations
import time

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from loguru import logger


class LoggingCallback(TrainerCallback):
    """
    Logs training metrics to loguru at each logging step.
    Complements (not replaces) the built-in HuggingFace progress bar.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs is None:
            return
        step = state.global_step
        epoch = round(state.epoch or 0, 2)
        loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")
        lr = logs.get("learning_rate")

        parts = [f"step={step}", f"epoch={epoch}"]
        if loss is not None:
            parts.append(f"train_loss={loss:.4f}")
        if eval_loss is not None:
            parts.append(f"eval_loss={eval_loss:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")

        logger.info("Training | " + " | ".join(parts))

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Training started.")

    def on_train_end(self, args, state, control, **kwargs):
        logger.info(f"Training ended. Best eval loss: {state.best_metric}")


class GPUMetricsCallback(TrainerCallback):
    """
    Logs GPU memory usage and timing metrics at each logging step.

    Metrics are injected into the HuggingFace `logs` dict so the built-in
    WandbCallback picks them up automatically — no separate wandb.log() needed.

    Logged metrics:
      gpu/memory_allocated_gb   — currently allocated GPU memory
      gpu/memory_reserved_gb    — total reserved by the caching allocator
      gpu/peak_memory_gb        — peak allocated since training start
      gpu/utilization_pct       — allocated / reserved (cache pressure indicator)
      timing/epoch_duration_s   — wall-clock seconds for the current epoch
      timing/total_elapsed_s    — wall-clock seconds since training start
      timing/samples_per_second — training throughput
    """

    def __init__(self):
        super().__init__()
        self._train_start: float = 0.0
        self._epoch_start: float = 0.0
        self._step_start: float = 0.0
        self._samples_seen: int = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start = time.perf_counter()
        self._epoch_start = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU tracking active: {name} | {total:.1f} GB VRAM")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_duration = time.perf_counter() - self._epoch_start
        epoch_num = int(state.epoch or 0)
        logger.info(f"Epoch {epoch_num} completed in {epoch_duration:.1f}s ({epoch_duration / 60:.1f}min)")

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info(f"  Peak GPU memory after epoch {epoch_num}: {peak:.2f} GB")

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.perf_counter()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs is None:
            return

        elapsed = time.perf_counter() - self._train_start

        # Timing metrics (always available)
        logs["timing/total_elapsed_s"] = round(elapsed, 1)
        logs["timing/epoch_elapsed_s"] = round(time.perf_counter() - self._epoch_start, 1)

        # Throughput: use global_step × batch_size × grad_accum as samples processed
        if elapsed > 0 and state.global_step > 0:
            effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
            total_samples = state.global_step * effective_batch
            logs["timing/samples_per_second"] = round(total_samples / elapsed, 2)

        # GPU metrics (only on CUDA)
        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

        logs["gpu/memory_allocated_gb"] = round(allocated, 3)
        logs["gpu/memory_reserved_gb"] = round(reserved, 3)
        logs["gpu/peak_memory_gb"] = round(peak, 3)

        # Utilization: how much of the reserved cache is actually in use
        if reserved > 0:
            logs["gpu/utilization_pct"] = round((allocated / reserved) * 100, 1)

    def on_train_end(self, args, state, control, **kwargs):
        total = time.perf_counter() - self._train_start
        logger.info(f"Total training time: {total:.1f}s ({total / 60:.1f}min)")

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"Peak GPU memory: {peak:.2f} GB / {total_vram:.1f} GB ({peak / total_vram * 100:.0f}% used)")
