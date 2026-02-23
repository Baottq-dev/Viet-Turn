"""
Collate functions for VAPDataset batching.
"""

import torch
from typing import Dict, List


def vap_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for VAPDataset.

    Pads audio waveforms to the longest in the batch.
    Labels and VA matrices should already be same length (fixed window).

    Returns:
        - audio_waveform: (B, max_samples) padded
        - audio_lengths: (B,) actual sample lengths
        - vap_labels: (B, window_frames)
        - va_matrix: (B, 2, window_frames)
        - texts: list of str (length B)
        - text_snapshots: list of list (length B)
        - file_ids: list of str (length B)
    """
    # Audio: pad to max length
    audio_list = [item["audio_waveform"] for item in batch]
    audio_lengths = torch.tensor([a.shape[0] for a in audio_list])
    max_len = audio_lengths.max().item()

    audio_padded = torch.zeros(len(batch), max_len)
    for i, a in enumerate(audio_list):
        audio_padded[i, :a.shape[0]] = a

    # Labels and VA matrix: stack (should be same size due to fixed window)
    vap_labels = torch.stack([item["vap_labels"] for item in batch])
    va_matrix = torch.stack([item["va_matrix"] for item in batch])

    # Text: keep as lists
    texts = [item["text"] for item in batch]
    text_snapshots = [item["text_snapshots"] for item in batch]
    file_ids = [item["file_id"] for item in batch]

    return {
        "audio_waveform": audio_padded,
        "audio_lengths": audio_lengths,
        "vap_labels": vap_labels,
        "va_matrix": va_matrix,
        "texts": texts,
        "text_snapshots": text_snapshots,
        "file_ids": file_ids,
    }
