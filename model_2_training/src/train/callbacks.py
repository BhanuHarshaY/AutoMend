"""
callbacks.py

Custom training callbacks for Model 2 SFT.

Currently provides:
  - LoggingCallback: logs train/eval loss to loguru at each logging step
  - WandbStandardCallback: remaps HuggingFace metric keys to a consistent
    naming convention that matches MLX/mlx-lm's W&B output, so all runs
    (CUDA, MPS, CPU) appear with the same metric names in the W&B dashboard.
  - EarlyStoppingCallback is handled natively by HuggingFace Trainer via
    load_best_model_at_end + metric_for_best_model in TrainingArguments.

Standardized metric names (used by both CUDA and MPS):
    train_loss              ← HF: loss          | MLX: train_loss
    val_loss                ← HF: eval_loss     | MLX: val_loss
    train_learning_rate     ← HF: learning_rate | MLX: (not logged)
    val_runtime             ← HF: eval_runtime  | MLX: val_time
    val_samples_per_second  ← HF: eval_samples_per_second
    val_steps_per_second    ← HF: eval_steps_per_second
"""

from __future__ import annotations
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from loguru import logger

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Maps HuggingFace Trainer metric keys → standardized keys that match MLX output
_HF_TO_STANDARD: dict[str, str] = {
    "loss":                      "train_loss",
    "eval_loss":                 "val_loss",
    "learning_rate":             "train_learning_rate",
    "eval_runtime":              "val_runtime",
    "eval_samples_per_second":   "val_samples_per_second",
    "eval_steps_per_second":     "val_steps_per_second",
}


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


class WandbStandardCallback(TrainerCallback):
    """
    Logs HuggingFace Trainer metrics to W&B using standardized key names
    that match MLX/mlx-lm's naming convention.

    The HuggingFace Trainer's built-in WandbCallback logs metrics like
    eval/loss, eval/runtime, etc. This callback additionally logs the same
    values under train_loss, val_loss, val_runtime, etc. so that CUDA and
    MPS runs are directly comparable in the W&B dashboard.

    This callback is additive — it does not replace the built-in WandbCallback.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if not _WANDB_AVAILABLE or logs is None:
            return
        if not state.is_world_process_zero:
            return

        standard = {}
        for hf_key, std_key in _HF_TO_STANDARD.items():
            if hf_key in logs:
                standard[std_key] = logs[hf_key]

        if standard:
            try:
                wandb.log(standard, step=state.global_step)
            except Exception as e:
                logger.warning(f"WandbStandardCallback: failed to log metrics: {e}")
