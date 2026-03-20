"""
paths.py

Centralized path management for the Model 2 training pipeline.

Provides a single source of truth for all standard directory and file paths,
derived from the model_2_training/ root. All paths are returned as
pathlib.Path objects.

Usage:
    from model_2_training.src.utils.paths import Paths

    p = Paths(m2_root)
    print(p.train_split)     # model_2_training/data/splits/train.jsonl
    print(p.checkpoints_dir) # model_2_training/outputs/checkpoints/
"""

from __future__ import annotations
from pathlib import Path


class Paths:
    """
    Centralized path resolver for Model 2 training.

    All paths are derived from the model_2_training/ root directory.
    No environment variables or hardcoded strings — everything is relative
    to m2_root so the package works in any location.
    """

    def __init__(self, m2_root: str | Path) -> None:
        """
        Args:
            m2_root: Absolute path to the model_2_training/ directory.
        """
        self.root = Path(m2_root).resolve()

    # --- Config paths ---

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def data_config(self) -> Path:
        return self.configs_dir / "data" / "track_b_chatml.yaml"

    @property
    def model_config(self) -> Path:
        return self.configs_dir / "model" / "qwen_baseline.yaml"

    @property
    def train_config(self) -> Path:
        return self.configs_dir / "train" / "qlora_sft.yaml"

    @property
    def eval_config(self) -> Path:
        return self.configs_dir / "eval" / "json_eval.yaml"

    # --- Data paths ---

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def splits_dir(self) -> Path:
        return self.data_dir / "splits"

    @property
    def train_split(self) -> Path:
        return self.splits_dir / "train.jsonl"

    @property
    def val_split(self) -> Path:
        return self.splits_dir / "val.jsonl"

    @property
    def test_split(self) -> Path:
        return self.splits_dir / "test.jsonl"

    @property
    def split_summary(self) -> Path:
        return self.splits_dir / "split_summary.json"

    # --- Output paths ---

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.outputs_dir / "checkpoints"

    @property
    def best_model_dir(self) -> Path:
        return self.checkpoints_dir / "best_model"

    @property
    def predictions_dir(self) -> Path:
        return self.outputs_dir / "predictions"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def val_reports_dir(self) -> Path:
        return self.reports_dir / "val"

    @property
    def test_reports_dir(self) -> Path:
        return self.reports_dir / "test"

    @property
    def logs_dir(self) -> Path:
        return self.outputs_dir / "logs"

    @property
    def runs_dir(self) -> Path:
        return self.outputs_dir / "runs"

    # --- Artifact path (from AutoMend data pipeline) ---

    @property
    def automend_root(self) -> Path:
        return self.root.parent

    @property
    def track_b_artifact(self) -> Path:
        return self.automend_root / "data" / "processed" / "track_B_combined.jsonl"

    def ensure_output_dirs(self) -> None:
        """Create all standard output directories if they don't exist."""
        for d in [
            self.splits_dir,
            self.checkpoints_dir,
            self.predictions_dir,
            self.val_reports_dir,
            self.test_reports_dir,
            self.logs_dir,
            self.runs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Paths(root={self.root})"
