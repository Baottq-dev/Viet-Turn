"""
Audio utilities for MM-VAP-VI.

Handles loading audio files and extracting features via Wav2Vec2/WavLM.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union


def load_audio(
    path: Union[str, Path],
    sample_rate: int = 16000,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample if needed.

    Args:
        path: Path to audio file.
        sample_rate: Target sample rate.
        mono: If True, convert to mono by averaging channels.

    Returns:
        waveform: (1, num_samples) tensor.
        sample_rate: Actual sample rate after resampling.
    """
    waveform, sr = torchaudio.load(str(path))

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    return waveform, sample_rate


def load_audio_segment(
    path: Union[str, Path],
    start_sec: float,
    end_sec: float,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Load a segment of audio.

    Args:
        path: Path to audio file.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        sample_rate: Target sample rate.

    Returns:
        waveform: (1, num_samples) tensor for the requested segment.
    """
    info = torchaudio.info(str(path))
    orig_sr = info.sample_rate

    frame_offset = int(start_sec * orig_sr)
    num_frames = int((end_sec - start_sec) * orig_sr)

    waveform, sr = torchaudio.load(
        str(path),
        frame_offset=frame_offset,
        num_frames=num_frames,
    )

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    return waveform


def get_audio_duration(path: Union[str, Path]) -> float:
    """Get audio duration in seconds."""
    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate


def samples_to_frames(num_samples: int, sample_rate: int = 16000, frame_hz: int = 50) -> int:
    """Convert number of audio samples to number of frames at given frame rate."""
    duration_sec = num_samples / sample_rate
    return int(duration_sec * frame_hz)


def frames_to_samples(num_frames: int, sample_rate: int = 16000, frame_hz: int = 50) -> int:
    """Convert number of frames to number of audio samples."""
    duration_sec = num_frames / frame_hz
    return int(duration_sec * sample_rate)


def seconds_to_frames(seconds: float, frame_hz: int = 50) -> int:
    """Convert seconds to frame index."""
    return int(seconds * frame_hz)


def frames_to_seconds(frames: int, frame_hz: int = 50) -> float:
    """Convert frame index to seconds."""
    return frames / frame_hz
