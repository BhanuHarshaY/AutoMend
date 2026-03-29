"""
wandb_logger.py

Weights & Biases experiment tracking for Model 1 training.

Gracefully degrades if wandb is not installed or WANDB_API_KEY is not set.
All public functions are safe to call regardless of wandb availability.
"""

from __future__ import annotations
import os
from pathlib import Path
from loguru import logger

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def _check_wandb() -> bool:
    """Return True if wandb is available and configured."""
    if not _WANDB_AVAILABLE:
        logger.warning("wandb not installed — skipping W&B logging")
        return False
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not set — skipping W&B logging")
        return False
    return True


def init_run(
    project: str = "automend-model1",
    run_name: str | None = None,
    config: dict | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    group: str | None = None,
) -> object | None:
    """Initialize a W&B run. Returns the run object or None."""
    if not _check_wandb():
        return None
    try:
        run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags or [],
            notes=notes,
            group=group,
            reinit=True,
        )
        logger.info(f"W&B run initialized: {run.url}")
        return run
    except Exception as e:
        logger.warning(f"W&B init failed: {e}")
        return None


def log_hyperparameters(params: dict) -> None:
    """Log hyperparameters to the active W&B run."""
    if not _check_wandb():
        return
    try:
        wandb.config.update(params, allow_val_change=True)
        logger.info(f"W&B: logged {len(params)} hyperparameters")
    except Exception as e:
        logger.warning(f"W&B log_hyperparameters failed: {e}")


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log a dict of metrics to the active W&B run."""
    if not _check_wandb():
        return
    try:
        wandb.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"W&B log_metrics failed: {e}")


def log_dataset_info(
    artifact_path: str | Path,
    split_sizes: dict[str, int],
) -> None:
    """Log dataset metadata to W&B config."""
    if not _check_wandb():
        return
    try:
        info = {
            "dataset/artifact_path": str(artifact_path),
            "dataset/train_size": split_sizes.get("train", 0),
            "dataset/val_size": split_sizes.get("val", 0),
            "dataset/test_size": split_sizes.get("test", 0),
        }
        wandb.config.update(info, allow_val_change=True)
        logger.info("W&B: dataset info logged")
    except Exception as e:
        logger.warning(f"W&B log_dataset_info failed: {e}")


def log_eval_metrics(metrics: dict, split: str = "val") -> None:
    """Log evaluation metrics under a split namespace."""
    if not _check_wandb():
        return
    try:
        loggable = {
            f"{split}/{k}": v
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        wandb.log(loggable)
        logger.info(f"W&B: logged {len(loggable)} eval metrics under '{split}/'")
    except Exception as e:
        logger.warning(f"W&B log_eval_metrics failed: {e}")


def log_confusion_matrix(y_true, y_pred, class_names: list[str]) -> None:
    """Log a confusion matrix as a W&B plot."""
    if not _check_wandb():
        return
    try:
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true.tolist() if hasattr(y_true, "tolist") else list(y_true),
                preds=y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred),
                class_names=class_names,
            )
        })
        logger.info("W&B: confusion matrix logged")
    except Exception as e:
        logger.warning(f"W&B log_confusion_matrix failed: {e}")


def log_checkpoint(checkpoint_path: str | Path, artifact_name: str = "model1-checkpoint") -> None:
    """Log a model checkpoint as a W&B Artifact."""
    if not _check_wandb():
        return
    try:
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_dir(str(checkpoint_path))
        wandb.log_artifact(artifact)
        logger.info(f"W&B: checkpoint logged as artifact '{artifact_name}'")
    except Exception as e:
        logger.warning(f"W&B log_checkpoint failed: {e}")


def log_bias_report(report: dict) -> None:
    """Log bias detection results to W&B."""
    if not _check_wandb():
        return
    try:
        flat = {}
        for slice_name, slice_metrics in report.get("slices", {}).items():
            for metric_name, value in slice_metrics.items():
                if isinstance(value, (int, float)):
                    flat[f"bias/{slice_name}/{metric_name}"] = value
        if report.get("disparity"):
            for metric, val in report["disparity"].items():
                flat[f"bias/disparity/{metric}"] = val
        wandb.log(flat)
        logger.info(f"W&B: logged bias report ({len(flat)} metrics)")
    except Exception as e:
        logger.warning(f"W&B log_bias_report failed: {e}")


def finish_run() -> None:
    """Finish the active W&B run cleanly."""
    if not _check_wandb():
        return
    try:
        wandb.finish()
        logger.info("W&B run finished")
    except Exception as e:
        logger.warning(f"W&B finish failed: {e}")
