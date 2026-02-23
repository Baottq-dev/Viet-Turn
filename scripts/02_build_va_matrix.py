#!/usr/bin/env python3
"""
02_build_va_matrix.py - Build Voice Activity Matrices

Converts RTTM diarization output to binary voice activity matrices
at 50fps (20ms per frame) for VAP label generation.

Usage:
    python scripts/02_build_va_matrix.py --rttm-dir data/rttm --audio-dir data/audio --output data/va_matrices
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.audio import get_audio_duration, seconds_to_frames


FRAME_HZ = 50  # 20ms per frame


def parse_rttm(rttm_path: Path) -> List[Dict]:
    """Parse RTTM file into segments."""
    segments = []
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9 and parts[0] == "SPEAKER":
                segments.append({
                    "speaker": parts[7],
                    "start": float(parts[3]),
                    "duration": float(parts[4]),
                    "end": float(parts[3]) + float(parts[4]),
                })
    return segments


def select_top_speakers(segments: List[Dict], n: int = 2) -> List[str]:
    """Select top N speakers by total speech duration."""
    speaker_dur = {}
    for seg in segments:
        spk = seg["speaker"]
        speaker_dur[spk] = speaker_dur.get(spk, 0) + seg["duration"]

    sorted_speakers = sorted(speaker_dur.items(), key=lambda x: -x[1])
    return [spk for spk, _ in sorted_speakers[:n]]


def build_va_matrix(
    segments: List[Dict],
    duration_sec: float,
    speaker_map: Dict[str, int],
    frame_hz: int = FRAME_HZ,
) -> np.ndarray:
    """
    Build binary voice activity matrix.

    Args:
        segments: List of diarization segments.
        duration_sec: Total audio duration in seconds.
        speaker_map: Mapping from speaker ID to index (0 or 1).
        frame_hz: Frame rate (50 = 20ms frames).

    Returns:
        va_matrix: (2, num_frames) binary numpy array.
    """
    num_frames = seconds_to_frames(duration_sec, frame_hz)
    va_matrix = np.zeros((2, num_frames), dtype=np.float32)

    for seg in segments:
        speaker = seg["speaker"]
        if speaker not in speaker_map:
            continue

        spk_idx = speaker_map[speaker]
        start_frame = seconds_to_frames(seg["start"], frame_hz)
        end_frame = seconds_to_frames(seg["end"], frame_hz)

        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)

        va_matrix[spk_idx, start_frame:end_frame] = 1.0

    return va_matrix


def compute_va_stats(va_matrix: np.ndarray) -> Dict:
    """Compute statistics for a VA matrix."""
    num_frames = va_matrix.shape[1]
    duration_sec = num_frames / FRAME_HZ

    s0_active = va_matrix[0].sum() / num_frames
    s1_active = va_matrix[1].sum() / num_frames
    overlap = ((va_matrix[0] > 0) & (va_matrix[1] > 0)).sum() / num_frames
    silence = ((va_matrix[0] == 0) & (va_matrix[1] == 0)).sum() / num_frames

    return {
        "num_frames": int(num_frames),
        "duration_sec": round(duration_sec, 2),
        "speaker_0_active_ratio": round(float(s0_active), 4),
        "speaker_1_active_ratio": round(float(s1_active), 4),
        "overlap_ratio": round(float(overlap), 4),
        "silence_ratio": round(float(silence), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Build voice activity matrices from RTTM")
    parser.add_argument("--rttm-dir", required=True, help="Directory with RTTM files")
    parser.add_argument("--audio-dir", required=True, help="Directory with audio files")
    parser.add_argument("--output", required=True, help="Output directory for VA matrices")
    parser.add_argument("--frame-hz", type=int, default=FRAME_HZ, help="Frame rate")
    args = parser.parse_args()

    rttm_dir = Path(args.rttm_dir)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rttm_files = sorted(rttm_dir.glob("*.rttm"))
    print(f"Found {len(rttm_files)} RTTM files")

    results = {}
    for i, rttm_path in enumerate(rttm_files):
        file_id = rttm_path.stem
        print(f"\n[{i+1}/{len(rttm_files)}] Processing {file_id}...")

        # Find corresponding audio
        audio_path = None
        for ext in [".wav", ".mp3", ".flac", ".m4a"]:
            candidate = audio_dir / f"{file_id}{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if audio_path is None:
            print(f"  WARNING: No audio file found for {file_id}, skipping")
            results[file_id] = {"status": "error", "error": "audio not found"}
            continue

        try:
            # Get audio duration
            duration_sec = get_audio_duration(audio_path)

            # Parse RTTM
            segments = parse_rttm(rttm_path)

            # Select top 2 speakers
            top_speakers = select_top_speakers(segments, n=2)
            speaker_map = {spk: idx for idx, spk in enumerate(top_speakers)}

            if len(top_speakers) < 2:
                print(f"  WARNING: Only {len(top_speakers)} speaker(s) found, padding with silence")
                # Pad speaker_map to always have 2 entries
                while len(speaker_map) < 2:
                    speaker_map[f"_empty_{len(speaker_map)}"] = len(speaker_map)

            # Build VA matrix
            va_matrix = build_va_matrix(segments, duration_sec, speaker_map, args.frame_hz)

            # Save as .pt
            output_path = output_dir / f"{file_id}.pt"
            torch.save(torch.from_numpy(va_matrix), output_path)

            # Stats
            stats = compute_va_stats(va_matrix)
            stats["speaker_map"] = speaker_map
            print(f"  {stats['num_frames']} frames, {stats['duration_sec']}s")
            print(f"  S0: {stats['speaker_0_active_ratio']:.1%}, S1: {stats['speaker_1_active_ratio']:.1%}")
            print(f"  Overlap: {stats['overlap_ratio']:.1%}, Silence: {stats['silence_ratio']:.1%}")

            results[file_id] = {"status": "ok", **stats}

        except Exception as e:
            print(f"  ERROR: {e}")
            results[file_id] = {"status": "error", "error": str(e)}

    # Summary
    summary_path = output_dir / "_va_matrix_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"\nDone: {ok}/{len(results)} files processed")


if __name__ == "__main__":
    main()
