"""
Frame-level evaluation metrics for VAP.

Tier 1: Measures quality of 256-class frame predictions.
Includes calibration metrics (ECE, Brier) and weighted F1.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
from sklearn.metrics import f1_score


def compute_frame_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -1,
) -> float:
    """
    Frame-level cross-entropy loss.

    Args:
        logits: (B, T, 256) raw model output.
        labels: (B, T) ground-truth class indices.
        ignore_index: Label to ignore.

    Returns:
        Cross-entropy loss (scalar).
    """
    mask = labels != ignore_index
    if not mask.any():
        return float("nan")

    logits_flat = logits[mask]
    labels_flat = labels[mask]
    return F.cross_entropy(logits_flat, labels_flat).item()


def compute_perplexity(ce_loss: float) -> float:
    """Convert CE loss to perplexity. PPL = exp(CE)."""
    if math.isnan(ce_loss):
        return float("nan")
    return math.exp(ce_loss)


def compute_topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 1,
    ignore_index: int = -1,
) -> float:
    """
    Top-k accuracy across valid frames.

    Args:
        logits: (B, T, 256)
        labels: (B, T)
        k: top-k
    """
    mask = labels != ignore_index
    if not mask.any():
        return float("nan")

    logits_flat = logits[mask]
    labels_flat = labels[mask]

    _, topk_preds = logits_flat.topk(k, dim=-1)
    correct = topk_preds.eq(labels_flat.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


def compute_weighted_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -1,
) -> float:
    """
    Weighted F1 score across all 256 classes.

    Weights each class by its frequency in the ground truth,
    handling the severe class imbalance in VAP labels.
    """
    mask = labels != ignore_index
    if not mask.any():
        return float("nan")

    preds = logits[mask].argmax(dim=-1).cpu().numpy()
    gt = labels[mask].cpu().numpy()
    return float(f1_score(gt, preds, average="weighted", zero_division=0))


def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
    ignore_index: int = -1,
) -> float:
    """
    Expected Calibration Error.

    Measures how well predicted confidence matches actual accuracy.
    A perfectly calibrated model has ECE = 0.

    Args:
        logits: (B, T, 256) raw model output.
        labels: (B, T) ground-truth class indices.
        n_bins: Number of confidence bins.
        ignore_index: Label to ignore.

    Returns:
        ECE (scalar, lower is better).
    """
    mask = labels != ignore_index
    if not mask.any():
        return float("nan")

    probs = F.softmax(logits[mask], dim=-1)
    confidences, preds = probs.max(dim=-1)
    gt = labels[mask]
    accuracies = (preds == gt).float()

    confidences = confidences.cpu().numpy()
    accuracies = accuracies.cpu().numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = in_bin.sum()
        if count > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += (count / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_brier_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 256,
    ignore_index: int = -1,
) -> float:
    """
    Brier Score — mean squared error between predicted probabilities and one-hot targets.

    Decomposes into calibration + refinement. Lower is better.
    Range: [0, 2] for multi-class.

    Args:
        logits: (B, T, 256) raw model output.
        labels: (B, T) ground-truth class indices.
        num_classes: Number of classes.
        ignore_index: Label to ignore.

    Returns:
        Brier score (scalar, lower is better).
    """
    mask = labels != ignore_index
    if not mask.any():
        return float("nan")

    probs = F.softmax(logits[mask], dim=-1)
    gt = labels[mask]

    # One-hot encode targets
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, gt.unsqueeze(1), 1.0)

    # Mean squared error across classes and samples
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean()
    return float(brier.item())


def compute_frame_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -1,
) -> Dict[str, float]:
    """
    Compute all frame-level metrics.

    Returns dict with: ce_loss, perplexity, top1_acc, top5_acc,
                       weighted_f1, ece, brier_score
    """
    ce = compute_frame_ce(logits, labels, ignore_index)
    return {
        "frame_ce": ce,
        "frame_perplexity": compute_perplexity(ce),
        "frame_top1_acc": compute_topk_accuracy(logits, labels, k=1, ignore_index=ignore_index),
        "frame_top5_acc": compute_topk_accuracy(logits, labels, k=5, ignore_index=ignore_index),
        "frame_weighted_f1": compute_weighted_f1(logits, labels, ignore_index=ignore_index),
        "frame_ece": compute_ece(logits, labels, ignore_index=ignore_index),
        "frame_brier": compute_brier_score(logits, labels, ignore_index=ignore_index),
    }
