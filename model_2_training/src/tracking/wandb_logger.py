"""
wandb_logger.py

Experiment tracking integration with Weights & Biases.

Responsibilities:
  - Initialize a W&B run with project/entity/config
  - Log hyperparameters at run start
  - Log metrics at each step (train loss, eval loss, JSON metrics)
  - Log dataset version info and split sizes
  - Log checkpoint paths as W&B artifacts
  - Log sample predictions for manual review in W&B UI
  - Finish the run cleanly
  - Gracefully degrade (log warning, continue) if wandb is not installed
    or WANDB_API_KEY is not set

All functions are safe to call even if wandb is unavailable.
"""

from __future__ import annotations
import os
from pathlib import Path
from loguru import logger

# Lazy import — wandb may not be installed in all environments
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def _check_wandb() -> bool:
    """Return True if wandb is available and WANDB_API_KEY is set."""
    if not _WANDB_AVAILABLE:
        logger.warning("wandb not installed — skipping W&B logging. Install with: pip install wandb")
        return False
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not set — skipping W&B logging.")
        return False
    return True


def init_run(
    project: str = "automend-model2",
    run_name: str | None = None,
    config: dict | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
) -> object | None:
    """
    Initialize a W&B run.

    Args:
        project: W&B project name.
        run_name: Optional human-readable run name.
        config: Dict of hyperparameters to log.
        tags: Optional list of tags for filtering in W&B UI.
        notes: Optional free-text notes about the run.

    Returns:
        The wandb.Run object if successful, else None.
    """
    if not _check_wandb():
        return None
    try:
        run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags or [],
            notes=notes,
            reinit=True,
        )
        logger.info(f"W&B run initialized: {run.url}")
        return run
    except Exception as e:
        logger.warning(f"W&B init failed: {e} — continuing without tracking.")
        return None


def log_hyperparameters(params: dict) -> None:
    """
    Log hyperparameters to the active W&B run.

    Args:
        params: Dict of hyperparameter names and values.
    """
    if not _check_wandb():
        return
    try:
        wandb.config.update(params, allow_val_change=True)
        logger.info(f"W&B: logged {len(params)} hyperparameters")
    except Exception as e:
        logger.warning(f"W&B log_hyperparameters failed: {e}")


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """
    Log a dict of metrics to the active W&B run.

    Args:
        metrics: Dict of metric_name -> value.
        step: Optional global step number.
    """
    if not _check_wandb():
        return
    try:
        wandb.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"W&B log_metrics failed: {e}")


def log_dataset_info(
    artifact_path: str | Path,
    split_sizes: dict[str, int],
    dvc_version: str | None = None,
) -> None:
    """
    Log dataset metadata to W&B config.

    Args:
        artifact_path: Path to the source JSONL artifact.
        split_sizes: Dict of split_name -> sample count.
        dvc_version: Optional DVC commit hash for data versioning.
    """
    if not _check_wandb():
        return
    try:
        info = {
            "dataset/artifact_path": str(artifact_path),
            "dataset/train_size": split_sizes.get("train", 0),
            "dataset/val_size": split_sizes.get("val", 0),
            "dataset/test_size": split_sizes.get("test", 0),
        }
        if dvc_version:
            info["dataset/dvc_version"] = dvc_version
        wandb.config.update(info, allow_val_change=True)
        logger.info("W&B: dataset info logged")
    except Exception as e:
        logger.warning(f"W&B log_dataset_info failed: {e}")


def log_json_metrics(metrics: dict, prefix: str = "eval") -> None:
    """
    Log JSON structural quality metrics with a prefix namespace.

    Args:
        metrics: Output of metrics_json.compute_metrics().
        prefix: Namespace prefix, e.g. "eval" or "test".
    """
    if not _check_wandb():
        return
    try:
        prefixed = {f"{prefix}/{k}": v for k, v in metrics.items()}
        wandb.log(prefixed)
        logger.info(f"W&B: logged {len(prefixed)} JSON metrics under '{prefix}/'")
    except Exception as e:
        logger.warning(f"W&B log_json_metrics failed: {e}")


def log_sample_predictions(predictions: list[dict], n: int = 20, step: int | None = None) -> None:
    """
    Log a sample of predictions as a W&B Table for inspection in the UI.

    Args:
        predictions: List of prediction dicts with "generated" and "reference".
        n: Number of samples to log.
        step: Optional global step.
    """
    if not _check_wandb():
        return
    try:
        table = wandb.Table(columns=["index", "generated", "reference"])
        for p in predictions[:n]:
            table.add_data(
                p.get("index", ""),
                p.get("generated", "")[:500],
                p.get("reference", "")[:500],
            )
        wandb.log({"sample_predictions": table}, step=step)
        logger.info(f"W&B: logged {min(n, len(predictions))} sample predictions as Table")
    except Exception as e:
        logger.warning(f"W&B log_sample_predictions failed: {e}")


def log_checkpoint(checkpoint_path: str | Path, artifact_name: str = "model-checkpoint") -> None:
    """
    Log a model checkpoint directory as a W&B Artifact.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        artifact_name: Name for the W&B artifact.
    """
    if not _check_wandb():
        return
    try:
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_dir(str(checkpoint_path))
        wandb.log_artifact(artifact)
        logger.info(f"W&B: checkpoint logged as artifact '{artifact_name}'")
    except Exception as e:
        logger.warning(f"W&B log_checkpoint failed: {e}")


def finish_run() -> None:
    """Finish the active W&B run cleanly."""
    if not _check_wandb():
        return
    try:
        wandb.finish()
        logger.info("W&B run finished.")
    except Exception as e:
        logger.warning(f"W&B finish failed: {e}")
