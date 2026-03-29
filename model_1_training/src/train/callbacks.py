"""
callbacks.py

Custom HuggingFace Trainer callbacks for Model 1 training.
"""

from __future__ import annotations
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from loguru import logger


class LogEpochMetricsCallback(TrainerCallback):
    """Log epoch-level metrics in a clean format after each evaluation."""

    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, metrics: dict = None, **kwargs):
        if metrics is None:
            return
        epoch = state.epoch or 0
        logger.info(f"{'=' * 50}")
        logger.info(f"Epoch {epoch:.0f} evaluation results:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        logger.info(f"{'=' * 50}")


class EarlyStoppingWithLogging(TrainerCallback):
    """
    Early stopping that logs when patience is exhausted.

    HF Trainer has built-in EarlyStoppingCallback, but this version
    provides more detailed logging about the countdown.
    """

    def __init__(self, patience: int = 3, metric: str = "eval_macro_f1"):
        self.patience = patience
        self.metric = metric
        self.best_value = float("-inf")
        self.wait = 0

    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, metrics: dict = None, **kwargs):
        if metrics is None or self.metric not in metrics:
            return

        current = metrics[self.metric]
        if current > self.best_value:
            self.best_value = current
            self.wait = 0
            logger.info(f"New best {self.metric}: {current:.4f}")
        else:
            self.wait += 1
            logger.info(
                f"{self.metric} did not improve ({current:.4f} vs best {self.best_value:.4f}). "
                f"Patience: {self.wait}/{self.patience}"
            )
            if self.wait >= self.patience:
                logger.warning(f"Early stopping triggered after {self.patience} epochs without improvement")
                control.should_training_stop = True
