"""YAML configuration loader with CLI override support."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(*cfgs: dict) -> dict[str, Any]:
    """Shallow-merge multiple config dicts (last wins)."""
    merged: dict[str, Any] = {}
    for c in cfgs:
        merged.update(c)
    return merged


def save_yaml(data: dict, path: str | Path) -> None:
    """Write a dict to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
