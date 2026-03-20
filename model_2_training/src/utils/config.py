"""
config.py

YAML configuration loading utilities for the Model 2 training pipeline.

Responsibilities:
  - Load a single YAML config file into a dict
  - Load and merge multiple config files (later files override earlier)
  - Validate that required keys are present
  - Provide typed accessors for common config values

All functions return plain Python dicts — no custom config objects.
This keeps the pipeline easy to serialize and log.
"""

from __future__ import annotations
from pathlib import Path
from loguru import logger

import yaml


def load_yaml(path: str | Path) -> dict:
    """
    Load a YAML config file and return it as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: if the file does not exist.
        yaml.YAMLError: if the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        logger.warning(f"Config file is empty: {path}")
        return {}
    logger.info(f"Loaded config: {path} ({len(cfg)} keys)")
    return cfg


def merge_configs(*configs: dict) -> dict:
    """
    Merge multiple config dicts. Later dicts override earlier ones (shallow merge).

    Args:
        *configs: Any number of config dicts to merge.

    Returns:
        Merged config dict.
    """
    merged = {}
    for cfg in configs:
        merged.update(cfg)
    return merged


def require_keys(cfg: dict, keys: list[str], context: str = "") -> None:
    """
    Assert that all required keys are present in a config dict.

    Args:
        cfg: Config dict to validate.
        keys: List of required key names.
        context: Optional label for error messages (e.g. "model config").

    Raises:
        KeyError: if any required key is missing.
    """
    missing = [k for k in keys if k not in cfg]
    if missing:
        label = f"[{context}] " if context else ""
        raise KeyError(f"{label}Missing required config keys: {missing}")


def load_all_configs(
    data_config_path: str | Path,
    model_config_path: str | Path,
    train_config_path: str | Path,
    eval_config_path: str | Path | None = None,
) -> tuple[dict, dict, dict, dict]:
    """
    Load all four standard configs in one call.

    Args:
        data_config_path: Path to track_b_chatml.yaml.
        model_config_path: Path to qwen_baseline.yaml.
        train_config_path: Path to qlora_sft.yaml.
        eval_config_path: Optional path to json_eval.yaml.

    Returns:
        (data_cfg, model_cfg, train_cfg, eval_cfg) — eval_cfg is {} if path not given.
    """
    data_cfg = load_yaml(data_config_path)
    model_cfg = load_yaml(model_config_path)
    train_cfg = load_yaml(train_config_path)
    eval_cfg = load_yaml(eval_config_path) if eval_config_path else {}
    return data_cfg, model_cfg, train_cfg, eval_cfg
