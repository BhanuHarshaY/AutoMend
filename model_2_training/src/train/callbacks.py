"""
callbacks.py

Custom training callbacks for Model 2 SFT.

Currently provides:
  - LoggingCallback: logs train/eval loss to loguru at each logging step
  - EarlyStoppingCallback is handled natively by HuggingFace Trainer via
    load_best_model_at_end + metric_for_best_model in TrainingArguments.
"""

from __future__ import annotations
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
