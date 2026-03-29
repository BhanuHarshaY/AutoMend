"""Device detection for training dispatch."""

from __future__ import annotations
import torch
from loguru import logger


def detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' based on available hardware."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA device detected: {name}")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple MPS device detected")
        return "mps"
    logger.info("No GPU detected — using CPU")
    return "cpu"
