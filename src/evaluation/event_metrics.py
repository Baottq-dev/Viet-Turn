"""
Event-level evaluation metrics for VAP.

Tier 2: Maps 256-class predictions to turn-taking events
(Shift, Hold, Backchannel) and computes Balanced Accuracy and F1.
"""

import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from src.utils.labels import NUM_BINS, NUM_CLASSES


def probs_to_p_now(probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert 256-class probs to per-speaker p_now (P active in next 200ms).

    Args:
        probs: (..., 256) softmax probabilities.

    Returns:
        p_now_0, p_now_1: (...,) probabilities for each speaker.
    """
    class_idx = torch.arange(NUM_CLASSES, device=probs.device)
    s0_mask = ((class_idx >> 0) & 1).bool()  # bit 0 = speaker 0, bin 0
    s1_mask = ((class_idx >> NUM_BINS) & 1).bool()  # bit 4 = speaker 1, bin 0

    p_now_0 = probs[..., s0_mask].sum(dim=-1)
    p_now_1 = probs[..., s1_mask].sum(dim=-1)
    return p_now_0, p_now_1


def classify_events(
    probs: torch.Tensor,
    va_matrix: torch.Tensor,
    shift_threshold: float = 0.5,
    bc_max_frames: int = 50,  # 1s at 50fps
    ignore_index: int = -1,
    labels: torch.Tensor = None,
) -> Dict[str, np.ndarray]:
    """
    Classify frames into turn-taking events based on VA ground truth.

    A frame is labeled as:
    - SHIFT: current speaker goes silent, other speaker starts (long)
    - HOLD: current speaker pauses but resumes
    - BACKCHANNEL: other speaker briefly active

    We determine ground truth events from VA matrix transitions,
    then evaluate model predictions against these.

    Args:
        probs: (T, 256) softmax probabilities.
        va_matrix: (2, T) binary ground-truth voice activity.
        shift_threshold: Threshold for p_now to classify as active.
        bc_max_frames: Maximum backchannel duration in frames.
        labels: (T,) VAP labels (optional, for masking invalid frames).

    Returns:
        Dict with gt_events and pred_events arrays.
    """
    T = probs.shape[0]
    device = probs.device

    p0, p1 = probs_to_p_now(probs)

    # Ground truth events from VA matrix
    # Detect speaker transitions
    s0_active = va_matrix[0].bool()
    s1_active = va_matrix[1].bool()

    # For simplicity: classify each frame based on the pattern
    # Assume speaker 0 is "current" initially, detect when control shifts
    gt_events = np.zeros(T, dtype=np.int64)  # 0=hold, 1=shift, 2=backchannel

    # Detect shift regions: s0 was active, then s1 becomes active for >bc_max_frames
    s0_np = va_matrix[0].cpu().numpy()
    s1_np = va_matrix[1].cpu().numpy()

    # Vectorized approach: find s1-only regions and classify by duration
    s1_only = (s0_np == 0) & (s1_np == 1)  # frames where only s1 is active

    # Find run lengths of s1-only regions
    # Compute run IDs: each contiguous block of s1_only gets a unique ID
    diff = np.diff(s1_only.astype(int), prepend=0)
    starts = np.where(diff == 1)[0]
    ends_diff = np.where(diff == -1)[0]

    # Handle case where last region extends to end
    if len(starts) > len(ends_diff):
        ends_diff = np.append(ends_diff, T)

    for s, e in zip(starts, ends_diff):
        duration = e - s
        if duration <= bc_max_frames:
            gt_events[s:e] = 2  # backchannel
        else:
            gt_events[s:e] = 1  # shift

    # s0 active frames stay 0 (hold), silence frames stay 0 (hold)

    # Predicted events from model probabilities
    pred_events = np.zeros(T, dtype=np.int64)
    p0_np = p0.cpu().numpy()
    p1_np = p1.cpu().numpy()

    for t in range(T):
        if p1_np[t] > shift_threshold and p0_np[t] < (1 - shift_threshold):
            pred_events[t] = 1  # predict shift
        elif p1_np[t] > 0.3 and p0_np[t] > 0.3:
            pred_events[t] = 2  # predict backchannel (both active briefly)
        else:
            pred_events[t] = 0  # hold

    # Mask invalid frames
    valid_mask = np.ones(T, dtype=bool)
    if labels is not None:
        valid_mask = labels.cpu().numpy() >= 0

    return {
        "gt_events": gt_events[valid_mask],
        "pred_events": pred_events[valid_mask],
        "p_shift": p1_np[valid_mask],
        "gt_shift": (gt_events[valid_mask] == 1).astype(np.float32),
    }


def compute_event_metrics(
    gt_events: np.ndarray,
    pred_events: np.ndarray,
    p_shift: np.ndarray = None,
    gt_shift: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute event-level metrics.

    Args:
        gt_events: (N,) ground-truth event labels (0=hold, 1=shift, 2=bc).
        pred_events: (N,) predicted event labels.
        p_shift: (N,) predicted P(shift) for AUC computation.
        gt_shift: (N,) binary shift labels.

    Returns:
        Dict with shift_hold_ba, bc_f1, predict_shift_auc.
    """
    metrics = {}

    # Shift/Hold Balanced Accuracy (binary: shift vs non-shift)
    gt_binary = (gt_events == 1).astype(int)
    pred_binary = (pred_events == 1).astype(int)

    if len(np.unique(gt_binary)) > 1:
        metrics["shift_hold_ba"] = balanced_accuracy_score(gt_binary, pred_binary)
    else:
        metrics["shift_hold_ba"] = float("nan")

    # Backchannel F1
    gt_bc = (gt_events == 2).astype(int)
    pred_bc = (pred_events == 2).astype(int)

    if gt_bc.sum() > 0:
        metrics["bc_f1"] = f1_score(gt_bc, pred_bc, zero_division=0)
    else:
        metrics["bc_f1"] = float("nan")

    # Predict-Shift AUC
    if p_shift is not None and gt_shift is not None and len(np.unique(gt_shift)) > 1:
        metrics["predict_shift_auc"] = roc_auc_score(gt_shift, p_shift)
    else:
        metrics["predict_shift_auc"] = float("nan")

    # 3-class balanced accuracy
    if len(np.unique(gt_events)) > 1:
        metrics["event_3class_ba"] = balanced_accuracy_score(gt_events, pred_events)
    else:
        metrics["event_3class_ba"] = float("nan")

    return metrics
