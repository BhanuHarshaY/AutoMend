"""
artifact_logger.py

Logs training artifacts (configs, metrics, checkpoints, reports) to local disk
in a structured, run-indexed format.

This is the local-disk complement to wandb_logger.py. It ensures every run's
outputs are fully reproducible and traceable even without a W&B account.

Each run gets a unique directory under outputs/runs/<run_id>/ containing:
  - run_config_snapshot.json   (all configs merged)
  - split_summary.json         (data split info)
  - metrics_val.json           (validation metrics)
  - metrics_test.json          (test metrics, if run)
  - run_metadata.json          (timestamps, paths, versions)
"""

from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path
from loguru import logger


def _now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(runs_root: str | Path, run_id: str | None = None) -> Path:
    """
    Create a unique directory for this training run.

    Args:
        runs_root: Base directory for all runs (e.g. outputs/runs/).
        run_id: Optional explicit ID. Defaults to timestamp.

    Returns:
        Path to the created run directory.
    """
    runs_root = Path(runs_root)
    run_id = run_id or f"run_{_now_str()}"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run directory created: {run_dir}")
    return run_dir


def save_run_metadata(
    run_dir: Path,
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    extra: dict | None = None,
) -> Path:
    """
    Save a merged metadata file for the run including all configs and timestamps.

    Args:
        run_dir: Run output directory.
        data_cfg: Data config dict.
        model_cfg: Model config dict.
        train_cfg: Train config dict.
        extra: Optional extra fields (e.g. wandb_run_url, checkpoint_path).

    Returns:
        Path to the saved metadata file.
    """
    metadata = {
        "run_id": run_dir.name,
        "timestamp": datetime.now().isoformat(),
        "data_config": data_cfg,
        "model_config": model_cfg,
        "train_config": train_cfg,
    }
    if extra:
        metadata.update(extra)

    path = run_dir / "run_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Run metadata saved → {path}")
    return path


def save_split_summary(split_summary: dict, run_dir: Path) -> Path:
    """Copy the split summary into the run directory for traceability."""
    path = run_dir / "split_summary.json"
    with open(path, "w") as f:
        json.dump(split_summary, f, indent=2)
    logger.info(f"Split summary logged → {path}")
    return path


def save_val_metrics(metrics: dict, run_dir: Path) -> Path:
    """Save validation metrics to the run directory."""
    path = run_dir / "metrics_val.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Validation metrics logged → {path}")
    return path


def save_test_metrics(metrics: dict, run_dir: Path) -> Path:
    """Save final test metrics to the run directory."""
    path = run_dir / "metrics_test.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Test metrics logged → {path}")
    return path


def copy_report(report_path: str | Path, run_dir: Path) -> Path:
    """Copy an evaluation report file into the run directory."""
    report_path = Path(report_path)
    dest = run_dir / report_path.name
    shutil.copy2(report_path, dest)
    logger.info(f"Report copied → {dest}")
    return dest


def log_run_summary(run_dir: Path, val_metrics: dict, test_metrics: dict | None = None) -> None:
    """
    Print a concise summary of the run to the logger.

    Args:
        run_dir: Path to run directory.
        val_metrics: Validation metrics dict.
        test_metrics: Optional test metrics dict.
    """
    logger.info("=" * 60)
    logger.info(f"RUN SUMMARY — {run_dir.name}")
    logger.info("-" * 60)
    logger.info("Validation metrics:")
    for k, v in val_metrics.items():
        logger.info(f"  {k:<30} {v}")
    if test_metrics:
        logger.info("Test metrics:")
        for k, v in test_metrics.items():
            logger.info(f"  {k:<30} {v}")
    logger.info(f"Artifacts: {run_dir}")
    logger.info("=" * 60)
