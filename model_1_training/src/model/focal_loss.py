"""
focal_loss.py

Focal Loss for multi-class classification with extreme class imbalance.

Track A data is ~99% Normal (class 0). Standard cross-entropy would converge
to always predicting Normal. Focal Loss down-weights well-classified samples
and focuses training on hard-to-classify anomaly instances.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples).
        alpha: Per-class weights tensor, or None for uniform.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes) raw logits.
            targets: (batch_size,) integer class labels.

        Returns:
            Scalar loss (if reduction='mean' or 'sum') or per-sample loss.
        """
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        pt = (probs * targets_one_hot).sum(dim=-1)
        focal_weight = (1.0 - pt) ** self.gamma

        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(
    label_distribution: Dict[int, int],
    num_classes: int = 7,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a label distribution.

    Args:
        label_distribution: {label_int: count}.
        num_classes: Total number of classes.
        smoothing: Smoothing factor to avoid division by zero for missing classes.

    Returns:
        Tensor of shape (num_classes,) with normalized weights.
    """
    total = sum(label_distribution.values())
    weights = torch.zeros(num_classes)

    for cls in range(num_classes):
        count = label_distribution.get(cls, 0) + smoothing
        weights[cls] = total / (num_classes * count)

    weights = weights / weights.sum() * num_classes

    logger.info(f"Computed class weights: {weights.tolist()}")
    return weights
