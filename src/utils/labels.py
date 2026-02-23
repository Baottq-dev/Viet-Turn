"""
VAP Label Encoding/Decoding Utilities.

Voice Activity Projection (VAP) represents future voice activity as a 256-class
classification problem. For each frame at 50fps (20ms):
  - Look ahead 2 seconds into the future
  - Divide into 4 bins per speaker: [0-200ms, 200-600ms, 600-1200ms, 1200-2000ms]
  - For each bin, compute if speaker is active > 50% of the bin duration
  - 2 speakers × 4 bins = 8 binary bits → 256 possible classes

Bin boundaries at 50fps (frame counts from current frame):
  Bin 0: frames 1-10    (0-200ms)
  Bin 1: frames 11-30   (200-600ms)
  Bin 2: frames 31-60   (600-1200ms)
  Bin 3: frames 61-100  (1200-2000ms)
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Dict


# Bin boundaries in frames at 50fps
BIN_FRAMES = [
    (1, 10),    # 0-200ms
    (11, 30),   # 200-600ms
    (31, 60),   # 600-1200ms
    (61, 100),  # 1200-2000ms
]

NUM_BINS = len(BIN_FRAMES)
NUM_SPEAKERS = 2
NUM_CLASSES = 2 ** (NUM_BINS * NUM_SPEAKERS)  # 256
FRAME_HZ = 50
FRAME_MS = 1000 / FRAME_HZ  # 20ms
LOOKAHEAD_FRAMES = 100  # 2 seconds at 50fps


def encode_vap_labels(
    va_matrix: np.ndarray,
    bin_frames: List[Tuple[int, int]] = BIN_FRAMES,
    activity_threshold: float = 0.5,
) -> np.ndarray:
    """
    Generate VAP labels from a voice activity matrix.

    Args:
        va_matrix: (2, num_frames) binary voice activity matrix.
                   va_matrix[s, t] = 1 if speaker s is active at frame t.
        bin_frames: List of (start_offset, end_offset) for each bin.
        activity_threshold: Fraction of bin that must be active to count as 1.

    Returns:
        labels: (num_frames,) array of class indices [0, 255].
                Frames where the full lookahead window is not available get label -1.
    """
    num_speakers, num_frames = va_matrix.shape
    assert num_speakers == NUM_SPEAKERS, f"Expected {NUM_SPEAKERS} speakers, got {num_speakers}"

    num_bins = len(bin_frames)
    max_lookahead = bin_frames[-1][1]  # Last bin end offset
    valid_frames = num_frames - max_lookahead

    labels = np.full(num_frames, -1, dtype=np.int64)

    if valid_frames <= 0:
        return labels

    # Vectorized: compute bin activity for all valid frames at once
    for b, (start_off, end_off) in enumerate(bin_frames):
        bin_length = end_off - start_off + 1
        threshold_count = int(np.ceil(activity_threshold * bin_length))

        for s in range(num_speakers):
            # Build activity counts using cumsum for efficiency
            cumsum = np.cumsum(np.concatenate([[0], va_matrix[s]]))
            # For each valid frame t, sum of activity in [t+start_off, t+end_off]
            frame_indices = np.arange(valid_frames)
            bin_activity = (
                cumsum[frame_indices + end_off + 1] - cumsum[frame_indices + start_off]
            )
            active = (bin_activity >= threshold_count).astype(np.int64)

            # Bit position: speaker 0 bins are bits 0-3, speaker 1 bins are bits 4-7
            bit_pos = s * num_bins + b
            if b == 0 and s == 0:
                labels[:valid_frames] = active << bit_pos
            else:
                labels[:valid_frames] |= active << bit_pos

    return labels


def encode_vap_labels_torch(
    va_matrix: torch.Tensor,
    bin_frames: List[Tuple[int, int]] = BIN_FRAMES,
    activity_threshold: float = 0.5,
) -> torch.Tensor:
    """
    PyTorch version of encode_vap_labels for GPU acceleration.

    Args:
        va_matrix: (2, num_frames) binary voice activity tensor.
        bin_frames: List of (start_offset, end_offset) for each bin.
        activity_threshold: Fraction of bin that must be active to count as 1.

    Returns:
        labels: (num_frames,) tensor of class indices [0, 255]. Invalid frames = -1.
    """
    num_speakers, num_frames = va_matrix.shape
    num_bins = len(bin_frames)
    max_lookahead = bin_frames[-1][1]
    valid_frames = num_frames - max_lookahead

    labels = torch.full((num_frames,), -1, dtype=torch.long, device=va_matrix.device)

    if valid_frames <= 0:
        return labels

    labels[:valid_frames] = 0

    for b, (start_off, end_off) in enumerate(bin_frames):
        bin_length = end_off - start_off + 1
        threshold_count = int(np.ceil(activity_threshold * bin_length))

        for s in range(num_speakers):
            # Cumulative sum for efficient windowed summation
            padded = torch.cat([torch.zeros(1, device=va_matrix.device), va_matrix[s].float()])
            cumsum = padded.cumsum(0)

            frame_idx = torch.arange(valid_frames, device=va_matrix.device)
            bin_activity = cumsum[frame_idx + end_off + 1] - cumsum[frame_idx + start_off]
            active = (bin_activity >= threshold_count).long()

            bit_pos = s * num_bins + b
            labels[:valid_frames] |= active << bit_pos

    return labels


def decode_vap_labels(class_idx: np.ndarray) -> np.ndarray:
    """
    Decode VAP class indices back to binary voice activity predictions.

    Args:
        class_idx: (...,) array of class indices [0, 255].

    Returns:
        predictions: (..., 2, 4) binary array.
                     predictions[..., s, b] = 1 if speaker s is predicted active in bin b.
    """
    shape = class_idx.shape
    flat = class_idx.flatten().astype(np.int64)
    result = np.zeros((flat.shape[0], NUM_SPEAKERS, NUM_BINS), dtype=np.int64)

    for s in range(NUM_SPEAKERS):
        for b in range(NUM_BINS):
            bit_pos = s * NUM_BINS + b
            result[:, s, b] = (flat >> bit_pos) & 1

    return result.reshape(*shape, NUM_SPEAKERS, NUM_BINS)


def decode_vap_labels_torch(class_idx: torch.Tensor) -> torch.Tensor:
    """
    PyTorch version of decode_vap_labels.

    Args:
        class_idx: (...,) tensor of class indices [0, 255].

    Returns:
        predictions: (..., 2, 4) binary tensor.
    """
    shape = class_idx.shape
    flat = class_idx.flatten().long()
    result = torch.zeros(flat.shape[0], NUM_SPEAKERS, NUM_BINS, dtype=torch.long, device=class_idx.device)

    for s in range(NUM_SPEAKERS):
        for b in range(NUM_BINS):
            bit_pos = s * NUM_BINS + b
            result[:, s, b] = (flat >> bit_pos) & 1

    return result.reshape(*shape, NUM_SPEAKERS, NUM_BINS)


def vap_probs_to_p_now(
    probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert 256-class VAP probabilities to p_now for each speaker.

    p_now[s] = probability that speaker s is active in the immediate future (bin 0: 0-200ms).

    Args:
        probs: (B, T, 256) softmax probabilities over VAP classes.

    Returns:
        p_now_0: (B, T) probability speaker 0 is active in next 200ms.
        p_now_1: (B, T) probability speaker 1 is active in next 200ms.
    """
    # Speaker 0 bin 0 is bit 0. Classes where bit 0 = 1 mean speaker 0 is active in bin 0.
    # Speaker 1 bin 0 is bit 4. Classes where bit 4 = 1 mean speaker 1 is active in bin 0.
    class_indices = torch.arange(NUM_CLASSES, device=probs.device)

    # Mask for classes where speaker 0 is active in bin 0
    s0_bin0_mask = ((class_indices >> 0) & 1).bool()  # bit 0
    # Mask for classes where speaker 1 is active in bin 0
    s1_bin0_mask = ((class_indices >> NUM_BINS) & 1).bool()  # bit 4

    p_now_0 = probs[:, :, s0_bin0_mask].sum(dim=-1)
    p_now_1 = probs[:, :, s1_bin0_mask].sum(dim=-1)

    return p_now_0, p_now_1


def vap_to_events(
    probs: torch.Tensor,
    current_speaker: int = 0,
    shift_threshold: float = 0.5,
    bc_threshold: float = 0.3,
    min_silence_frames: int = 15,  # 300ms at 50fps
) -> List[Dict]:
    """
    Convert VAP probabilities to turn-taking events.

    Events:
    - SHIFT: current speaker yields, other speaker takes over
    - HOLD: current speaker keeps the floor
    - BACKCHANNEL: other speaker gives brief feedback without taking the floor

    Args:
        probs: (T, 256) softmax probabilities for a single conversation.
        current_speaker: Index of the current speaker (0 or 1).
        shift_threshold: p_now threshold for detecting speaker change.
        bc_threshold: Threshold for backchannel detection.
        min_silence_frames: Minimum silence before a shift is confirmed.

    Returns:
        List of event dicts: [{type, start_frame, end_frame, confidence}, ...]
    """
    other_speaker = 1 - current_speaker
    T = probs.shape[0]
    class_indices = torch.arange(NUM_CLASSES, device=probs.device)

    # Compute p_now for each speaker (probability active in bin 0)
    curr_bit = current_speaker * NUM_BINS
    other_bit = other_speaker * NUM_BINS

    curr_active_mask = ((class_indices >> curr_bit) & 1).bool()
    other_active_mask = ((class_indices >> other_bit) & 1).bool()

    p_curr = probs[:, curr_active_mask].sum(dim=-1)  # (T,)
    p_other = probs[:, other_active_mask].sum(dim=-1)  # (T,)

    # Also check longer-term activity for backchannel vs shift distinction
    # Backchannel: other speaker active in bin 0 but NOT in bins 2-3
    other_long_bits = [other_speaker * NUM_BINS + b for b in [2, 3]]
    other_long_mask = torch.zeros(NUM_CLASSES, dtype=torch.bool, device=probs.device)
    for bit in other_long_bits:
        other_long_mask |= ((class_indices >> bit) & 1).bool()
    p_other_long = probs[:, other_long_mask].sum(dim=-1)  # (T,)

    events = []
    t = 0

    while t < T:
        # Detect when current speaker goes silent and other becomes active
        if p_curr[t] < (1 - shift_threshold) and p_other[t] > shift_threshold:
            start = t

            # Check if this is backchannel (short) or shift (long)
            if p_other_long[t] < bc_threshold:
                # Backchannel: other speaker briefly active
                end = t
                while end < T and p_other[end] > bc_threshold:
                    end += 1
                events.append({
                    "type": "BACKCHANNEL",
                    "start_frame": start,
                    "end_frame": end,
                    "confidence": float(p_other[start]),
                })
                t = end
            else:
                # Potential shift: check for sustained silence from current speaker
                silence_count = 0
                end = t
                while end < T and silence_count < min_silence_frames:
                    if p_curr[end] < (1 - shift_threshold):
                        silence_count += 1
                    else:
                        silence_count = 0
                    end += 1

                if silence_count >= min_silence_frames:
                    events.append({
                        "type": "SHIFT",
                        "start_frame": start,
                        "end_frame": end,
                        "confidence": float(p_other[start]),
                    })
                else:
                    events.append({
                        "type": "HOLD",
                        "start_frame": start,
                        "end_frame": end,
                        "confidence": float(p_curr[start]),
                    })
                t = end
        else:
            t += 1

    return events


def get_label_statistics(labels: np.ndarray) -> Dict:
    """
    Compute statistics for a set of VAP labels.

    Args:
        labels: (num_frames,) array of class indices.

    Returns:
        Dictionary with label distribution statistics.
    """
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]

    if len(valid_labels) == 0:
        return {"total_frames": len(labels), "valid_frames": 0}

    decoded = decode_vap_labels(valid_labels)
    # decoded shape: (valid_frames, 2, 4)

    stats = {
        "total_frames": len(labels),
        "valid_frames": int(valid_mask.sum()),
        "invalid_frames": int((~valid_mask).sum()),
        "unique_classes": int(len(np.unique(valid_labels))),
    }

    # Per-speaker, per-bin activity rates
    for s in range(NUM_SPEAKERS):
        for b in range(NUM_BINS):
            key = f"speaker{s}_bin{b}_active_rate"
            stats[key] = float(decoded[:, s, b].mean())

    # Most common classes
    unique, counts = np.unique(valid_labels, return_counts=True)
    top_k = min(10, len(unique))
    top_indices = np.argsort(-counts)[:top_k]
    stats["top_classes"] = [
        {"class": int(unique[i]), "count": int(counts[i]), "ratio": float(counts[i] / len(valid_labels))}
        for i in top_indices
    ]

    return stats
