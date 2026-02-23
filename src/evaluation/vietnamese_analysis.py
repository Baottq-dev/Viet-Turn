"""
Vietnamese-specific evaluation analysis.

Provides discourse marker impact analysis and per-dialect performance breakdown
for understanding model behavior on Vietnamese turn-taking patterns.
"""

import numpy as np
from typing import Dict, List, Optional

# Default Vietnamese discourse marker sets (subset of HuTuDetector markers)
DEFAULT_MARKER_SETS = {
    "yield": ["ạ", "nhé", "nhỉ", "nha", "ha", "hen", "nghen", "rồi đó", "vậy đó"],
    "hold": ["mà", "là", "thì", "nghĩa là", "tức là", "có nghĩa là"],
    "backchannel": ["ừ", "ờ", "ừm", "vâng", "dạ", "à", "ồ", "ừa"],
    "turn_request": ["này", "ơi", "nè", "cho hỏi", "cho mình hỏi"],
}


def analyze_marker_impact(
    events: List[Dict],
    p_shift: np.ndarray,
    threshold: float = 0.5,
    marker_sets: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """
    Compare model performance on events with/without discourse markers.

    Args:
        events: List of event dicts with keys:
            - "type": "shift" or "hold"
            - "text": transcript text near the event
            - "eval_frame": frame index to evaluate prediction
        p_shift: (T,) predicted P(shift) per frame.
        threshold: Decision threshold.
        marker_sets: Dict mapping category → list of markers.

    Returns:
        Dict with accuracy comparison and marker benefit delta.
    """
    if marker_sets is None:
        marker_sets = DEFAULT_MARKER_SETS

    with_marker = {"shift": [], "hold": []}
    without_marker = {"shift": [], "hold": []}

    all_markers = [m for markers in marker_sets.values() for m in markers]

    for event in events:
        text = event.get("text", "").lower()
        event_type = event.get("type", "hold")
        eval_frame = event.get("eval_frame", 0)

        if event_type not in ("shift", "hold"):
            continue
        if eval_frame >= len(p_shift):
            continue

        # Check if any marker appears in text (word-level matching)
        words = text.split()
        has_marker = any(m in words for m in all_markers)
        # Also check multi-word markers
        if not has_marker:
            has_marker = any(m in text for m in all_markers if " " in m)

        # Determine if prediction is correct
        pred_shift = p_shift[eval_frame] >= threshold
        correct = (pred_shift and event_type == "shift") or \
                  (not pred_shift and event_type == "hold")

        bucket = with_marker if has_marker else without_marker
        bucket[event_type].append(int(correct))

    def safe_mean(arr):
        return float(np.mean(arr)) if arr else float("nan")

    result = {
        "with_marker": {
            "shift_accuracy": safe_mean(with_marker["shift"]),
            "hold_accuracy": safe_mean(with_marker["hold"]),
            "shift_count": len(with_marker["shift"]),
            "hold_count": len(with_marker["hold"]),
        },
        "without_marker": {
            "shift_accuracy": safe_mean(without_marker["shift"]),
            "hold_accuracy": safe_mean(without_marker["hold"]),
            "shift_count": len(without_marker["shift"]),
            "hold_count": len(without_marker["hold"]),
        },
    }

    # Compute marker benefit (delta)
    shift_delta = safe_mean(with_marker["shift"]) - safe_mean(without_marker["shift"])
    hold_delta = safe_mean(with_marker["hold"]) - safe_mean(without_marker["hold"])

    result["marker_benefit"] = {
        "shift_delta": round(shift_delta, 4) if not np.isnan(shift_delta) else float("nan"),
        "hold_delta": round(hold_delta, 4) if not np.isnan(hold_delta) else float("nan"),
    }

    return result


def analyze_per_dialect(
    per_conversation_results: List[Dict],
) -> Dict[str, Dict]:
    """
    Break down metrics by Vietnamese dialect (Bắc/Trung/Nam).

    Args:
        per_conversation_results: List of dicts, each with:
            - "dialect": one of "bac", "trung", "nam", "mixed", or None
            - "shift_hold_ba": balanced accuracy for this conversation
            - "bc_f1": backchannel F1
            - "eot_latency_mean_ms": mean EOT latency

    Returns:
        Dict mapping dialect → aggregated metrics.
    """
    dialect_buckets: Dict[str, List[Dict]] = {}

    for conv in per_conversation_results:
        dialect = conv.get("dialect", "unknown") or "unknown"
        dialect_buckets.setdefault(dialect, []).append(conv)

    results = {}
    for dialect, convs in sorted(dialect_buckets.items()):
        metrics = {}
        for key in ["shift_hold_ba", "bc_f1", "eot_latency_mean_ms"]:
            values = [c[key] for c in convs if key in c and not np.isnan(c.get(key, float("nan")))]
            if values:
                metrics[key] = round(float(np.mean(values)), 4)
                metrics[f"{key}_std"] = round(float(np.std(values)), 4)
            else:
                metrics[key] = float("nan")
        metrics["n_conversations"] = len(convs)
        results[dialect] = metrics

    return results
