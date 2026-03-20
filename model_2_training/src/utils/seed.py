"""
seed.py

Global random seed management for reproducibility.

Sets seeds across Python, NumPy, PyTorch (CPU and CUDA), and optionally
HuggingFace Transformers. A single call to set_seed() is enough to make
the whole pipeline deterministic for a given seed value.
"""

from __future__ import annotations
import random
from loguru import logger


def set_seed(seed: int) -> None:
    """
    Set random seeds for full reproducibility.

    Covers: Python random, NumPy, PyTorch CPU, PyTorch CUDA (all GPUs),
    and HuggingFace transformers (if installed).

    Args:
        seed: Integer seed value. The guide default is 42.
    """
    random.seed(seed)
    logger.debug(f"Python random seed set to {seed}")

    try:
        import numpy as np
        np.random.seed(seed)
        logger.debug(f"NumPy random seed set to {seed}")
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Deterministic ops (slight perf cost — remove if speed is priority)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.debug(f"PyTorch seed set to {seed} (all devices)")
    except ImportError:
        pass

    try:
        import transformers
        transformers.set_seed(seed)
        logger.debug(f"HuggingFace transformers seed set to {seed}")
    except (ImportError, AttributeError):
        pass

    logger.info(f"Global seed set to {seed}")


def get_default_seed() -> int:
    """Return the project-wide default seed (42)."""
    return 42
