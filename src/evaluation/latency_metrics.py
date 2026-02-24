"""
Latency evaluation metrics for VAP.

Tier 3: Measures how quickly the model detects turn-taking events.
Includes VAQI (Voice Agent Quality Index) and Levenshtein EoT alignment.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_eot_latency(
    p_shift: np.ndarray,
    gt_shift_regions: List[Tuple[int, int]],
    threshold: float = 0.5,
    frame_hz: int = 50,
) -> Dict[str, float]:
    """
    Compute End-of-Turn latency.

    For each ground-truth shift region, find when the model first predicts
    P(shift) > threshold within or after the region.

    Args:
        p_shift: (T,) predicted P(shift) at each frame.
        gt_shift_regions: List of (start_frame, end_frame) for GT shifts.
        threshold: Decision threshold.
        frame_hz: Frame rate.

    Returns:
        Dict with mean/median/p95 latency in ms, and detection rate.
    """
    latencies = []
    detected = 0

    for start, end in gt_shift_regions:
        # Look for first frame with p_shift > threshold
        # Search from shift start to shift end + 2 seconds
        search_end = min(len(p_shift), end + 2 * frame_hz)

        found = False
        for t in range(start, search_end):
            if p_shift[t] > threshold:
                latency_frames = t - start
                latency_ms = latency_frames * (1000 / frame_hz)
                latencies.append(latency_ms)
                detected += 1
                found = True
                break

        if not found:
            latencies.append(float("inf"))

    if not gt_shift_regions:
        return {
            "eot_latency_mean_ms": float("nan"),
            "eot_latency_median_ms": float("nan"),
            "eot_latency_p95_ms": float("nan"),
            "detection_rate": float("nan"),
        }

    finite_latencies = [l for l in latencies if l != float("inf")]

    return {
        "eot_latency_mean_ms": float(np.mean(finite_latencies)) if finite_latencies else float("inf"),
        "eot_latency_median_ms": float(np.median(finite_latencies)) if finite_latencies else float("inf"),
        "eot_latency_p95_ms": float(np.percentile(finite_latencies, 95)) if finite_latencies else float("inf"),
        "detection_rate": detected / len(gt_shift_regions),
    }


def compute_fpr_at_thresholds(
    p_shift: np.ndarray,
    gt_shift: np.ndarray,
    thresholds_ms: List[float] = [100, 200, 300, 500, 1000],
    frame_hz: int = 50,
) -> Dict[str, float]:
    """
    Compute False Positive Rate at various latency thresholds.

    FPR = false positives / (false positives + true negatives)

    Args:
        p_shift: (T,) predicted P(shift).
        gt_shift: (T,) binary ground truth (1=shift region).
        thresholds_ms: List of time thresholds.
        frame_hz: Frame rate.

    Returns:
        Dict mapping threshold to FPR.
    """
    results = {}

    for thresh_ms in thresholds_ms:
        thresh_frames = int(thresh_ms / (1000 / frame_hz))

        # At this threshold: predict shift if p_shift > 0.5 within thresh_frames
        pred = (p_shift > 0.5).astype(float)

        # FPR on non-shift frames
        non_shift_mask = gt_shift == 0
        if non_shift_mask.sum() > 0:
            fp = pred[non_shift_mask].sum()
            fpr = fp / non_shift_mask.sum()
            results[f"fpr_at_{thresh_ms}ms"] = float(fpr)
        else:
            results[f"fpr_at_{thresh_ms}ms"] = float("nan")

    return results


def compute_mst_fpr_curve(
    p_shift: np.ndarray,
    gt_shift: np.ndarray,
    mst_range_ms: List[int] = None,
    frame_hz: int = 50,
) -> Dict:
    """
    Compute Minimum Silence Threshold vs FPR curve.

    MST: only trigger shift if P(shift) > threshold for at least MST consecutive frames.

    Args:
        p_shift: (T,) predicted P(shift).
        gt_shift: (T,) binary ground truth.
        mst_range_ms: List of MST values to evaluate.
        frame_hz: Frame rate.

    Returns:
        Dict with mst_values, fpr_values, and auc.
    """
    if mst_range_ms is None:
        mst_range_ms = list(range(0, 1100, 100))

    fpr_values = []
    non_shift_mask = gt_shift == 0

    for mst_ms in mst_range_ms:
        mst_frames = max(1, int(mst_ms / (1000 / frame_hz)))

        # Apply MST: predict shift only if p_shift > 0.5 for MST consecutive frames
        T = len(p_shift)
        pred = np.zeros(T)
        consecutive = 0

        for t in range(T):
            if p_shift[t] > 0.5:
                consecutive += 1
                if consecutive >= mst_frames:
                    pred[t] = 1
            else:
                consecutive = 0

        # FPR
        if non_shift_mask.sum() > 0:
            fpr = pred[non_shift_mask].sum() / non_shift_mask.sum()
        else:
            fpr = 0.0
        fpr_values.append(float(fpr))

    # AUC of MST-FPR curve (lower is better)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trapz(fpr_values, [ms / 1000 for ms in mst_range_ms]))

    return {
        "mst_values_ms": mst_range_ms,
        "fpr_values": fpr_values,
        "mst_fpr_auc": auc,
    }


def compute_vaqi(
    p_shift: np.ndarray,
    gt_shift_regions: List[Tuple[int, int]],
    gt_hold_regions: List[Tuple[int, int]],
    threshold: float = 0.5,
    frame_hz: int = 50,
    max_latency_ms: float = 2000.0,
) -> Dict[str, float]:
    """
    Compute Voice Agent Quality Index (VAQI).

    Adapts Deepgram's VAQI for offline evaluation:
        VAQI = 100 × (1 - [0.4×I + 0.4×M + 0.2×L])

    where:
        I = interruption rate (hold events misclassified as shift)
        M = missed response rate (shift events not detected within 1s)
        L = latency score (log-scaled median detection latency)

    Args:
        p_shift: (T,) predicted P(shift) per frame.
        gt_shift_regions: List of (start, end) for ground-truth shift regions.
        gt_hold_regions: List of (start, end) for ground-truth hold regions.
        threshold: Decision threshold for P(shift).
        frame_hz: Frame rate.
        max_latency_ms: Maximum latency for normalization.

    Returns:
        Dict with vaqi score and component breakdown.
    """
    # I: Interruption rate — hold regions where model predicts shift
    interruptions = 0
    for start, end in gt_hold_regions:
        # Check if model predicts shift anywhere in the hold region
        region = p_shift[start:end]
        if len(region) > 0 and np.any(region >= threshold):
            interruptions += 1
    I = interruptions / len(gt_hold_regions) if gt_hold_regions else 0.0

    # M: Missed response rate — shift events not detected within 1s
    missed = 0
    latencies = []
    search_window = frame_hz  # 1 second

    for start, end in gt_shift_regions:
        search_end = min(len(p_shift), end + search_window)
        detected = False
        for t in range(start, search_end):
            if p_shift[t] >= threshold:
                latency_ms = (t - start) * (1000.0 / frame_hz)
                latencies.append(latency_ms)
                detected = True
                break
        if not detected:
            missed += 1

    M = missed / len(gt_shift_regions) if gt_shift_regions else 0.0

    # L: Latency score (log-scaled)
    median_latency = float(np.median(latencies)) if latencies else max_latency_ms
    L = math.log(1 + median_latency) / math.log(1 + max_latency_ms)
    L = min(L, 1.0)

    vaqi = 100.0 * (1.0 - (0.4 * I + 0.4 * M + 0.2 * L))

    return {
        "vaqi": round(vaqi, 1),
        "vaqi_interruption_rate": round(I, 4),
        "vaqi_missed_response_rate": round(M, 4),
        "vaqi_latency_score": round(L, 4),
        "vaqi_median_latency_ms": round(median_latency, 1),
    }


def compute_eot_levenshtein(
    gt_eot_frames: List[int],
    pred_eot_frames: List[int],
    tolerance_frames: int = 25,
    frame_hz: int = 50,
) -> Dict[str, float]:
    """
    Sequence-based EoT evaluation using modified Levenshtein alignment.

    Aligns predicted [EoT] tokens to ground-truth [EoT] tokens using
    a modified edit distance where matches within tolerance are accepted.

    Metrics:
        - EoT Precision: fraction of predicted EoTs that match a GT EoT
        - EoT Recall: fraction of GT EoTs that were predicted
        - EoT F1
        - Mean position error (in ms) for matched pairs

    Args:
        gt_eot_frames: Sorted list of ground-truth EoT frame indices.
        pred_eot_frames: Sorted list of predicted EoT frame indices.
        tolerance_frames: Max distance (frames) for a match.
        frame_hz: Frame rate.

    Returns:
        Dict with precision, recall, F1, and mean position error.
    """
    if not gt_eot_frames or not pred_eot_frames:
        return {
            "eot_precision": 0.0 if pred_eot_frames else float("nan"),
            "eot_recall": 0.0 if gt_eot_frames else float("nan"),
            "eot_f1": 0.0,
            "eot_mean_position_error_ms": float("nan"),
        }

    gt = sorted(gt_eot_frames)
    pred = sorted(pred_eot_frames)

    # Greedy matching: for each GT EoT, find closest unmatched predicted EoT
    matched_gt = set()
    matched_pred = set()
    position_errors = []

    for i, g in enumerate(gt):
        best_j = None
        best_dist = tolerance_frames + 1
        for j, p in enumerate(pred):
            if j in matched_pred:
                continue
            dist = abs(g - p)
            if dist <= tolerance_frames and dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None:
            matched_gt.add(i)
            matched_pred.add(best_j)
            position_errors.append(best_dist * (1000.0 / frame_hz))

    precision = len(matched_pred) / len(pred) if pred else 0.0
    recall = len(matched_gt) / len(gt) if gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_error = float(np.mean(position_errors)) if position_errors else float("nan")

    return {
        "eot_precision": round(precision, 4),
        "eot_recall": round(recall, 4),
        "eot_f1": round(f1, 4),
        "eot_mean_position_error_ms": round(mean_error, 1),
    }
