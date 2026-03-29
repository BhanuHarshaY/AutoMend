"""Reproducibility utilities."""

from __future__ import annotations
import random

import numpy as np
import torch
from loguru import logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")
