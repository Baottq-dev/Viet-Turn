"""
VAPDataset - Sliding window dataset for MM-VAP-VI training.

Each sample is a window of audio + text + VAP labels from a conversation.
Windows are 20s with 5s stride by default, yielding 1000 frames at 50fps.
"""

import json
import bisect
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.audio import load_audio_segment, frames_to_seconds


class VAPDataset(Dataset):
    """
    Sliding window dataset for VAP training.

    Each item returns:
        - audio_waveform: (1, num_samples) raw audio for Wav2Vec2
        - vap_labels: (num_frames,) class indices [0-255], -1 for invalid
        - va_matrix: (2, num_frames) binary voice activity
        - text: str, cumulative text available at window start
        - text_snapshots: list of {frame, text} for within-window text updates
        - file_id: str
        - window_start_frame: int
    """

    def __init__(
        self,
        manifest_path: str,
        window_sec: float = 20.0,
        stride_sec: float = 5.0,
        frame_hz: int = 50,
        sample_rate: int = 16000,
        min_valid_ratio: float = 0.3,
    ):
        """
        Args:
            manifest_path: Path to manifest JSON (list of file entries).
            window_sec: Window duration in seconds.
            stride_sec: Stride between windows in seconds.
            frame_hz: Frame rate (50fps for VAP).
            sample_rate: Audio sample rate.
            min_valid_ratio: Minimum ratio of valid (non -1) labels in a window.
        """
        self.window_frames = int(window_sec * frame_hz)
        self.stride_frames = int(stride_sec * frame_hz)
        self.frame_hz = frame_hz
        self.sample_rate = sample_rate
        self.min_valid_ratio = min_valid_ratio

        # Load manifest
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

        # Pre-load metadata and build window index
        self.file_data = {}
        self.windows = []  # (file_idx, start_frame)
        self._build_index()

    def _build_index(self):
        """Build index of all valid windows across all files."""
        for file_idx, entry in enumerate(self.entries):
            file_id = entry["file_id"]

            # Load labels to get frame count
            labels = torch.load(entry["vap_label_path"], weights_only=True)
            num_frames = labels.shape[0]

            # Pre-load VA matrix to avoid repeated disk I/O in __getitem__
            va_matrix = torch.load(entry["va_matrix_path"], weights_only=True)

            # Load text alignment
            text_snapshots = []
            if entry.get("text_frames_path") and Path(entry["text_frames_path"]).exists():
                with open(entry["text_frames_path"], "r", encoding="utf-8") as f:
                    text_data = json.load(f)
                text_snapshots = text_data.get("text_snapshots", [])

            self.file_data[file_idx] = {
                "entry": entry,
                "num_frames": num_frames,
                "labels": labels,
                "va_matrix": va_matrix,
                "text_snapshots": text_snapshots,
            }

            # Generate windows
            start = 0
            while start + self.window_frames <= num_frames:
                # Check if window has enough valid labels
                window_labels = labels[start:start + self.window_frames]
                valid_ratio = (window_labels >= 0).float().mean().item()
                if valid_ratio >= self.min_valid_ratio:
                    self.windows.append((file_idx, start))
                start += self.stride_frames

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict:
        file_idx, start_frame = self.windows[idx]
        data = self.file_data[file_idx]
        entry = data["entry"]
        end_frame = start_frame + self.window_frames

        # Audio: load segment
        start_sec = frames_to_seconds(start_frame, self.frame_hz)
        end_sec = frames_to_seconds(end_frame, self.frame_hz)
        audio_waveform = load_audio_segment(
            entry["audio_path"],
            start_sec, end_sec,
            sample_rate=self.sample_rate,
        )  # (1, num_samples)

        # VAP labels for this window
        vap_labels = data["labels"][start_frame:end_frame]

        # VA matrix for this window (pre-loaded)
        va_matrix = data["va_matrix"][:, start_frame:end_frame]

        # Text: find the latest text snapshot at or before the window end
        text_snapshots = data["text_snapshots"]
        snapshot_frames = [s["frame"] for s in text_snapshots]

        # Text at window start (for initial PhoBERT encoding)
        snap_idx = bisect.bisect_right(snapshot_frames, start_frame) - 1
        text_at_start = text_snapshots[snap_idx]["text"] if snap_idx >= 0 else ""

        # Text snapshots within the window (for potential mid-window updates)
        window_snapshots = []
        for s in text_snapshots:
            if start_frame < s["frame"] <= end_frame:
                window_snapshots.append({
                    "relative_frame": s["frame"] - start_frame,
                    "text": s["text"],
                })

        return {
            "audio_waveform": audio_waveform.squeeze(0),  # (num_samples,)
            "vap_labels": vap_labels,        # (window_frames,)
            "va_matrix": va_matrix,          # (2, window_frames)
            "text": text_at_start,           # str
            "text_snapshots": window_snapshots,
            "file_id": entry["file_id"],
            "window_start_frame": start_frame,
        }
