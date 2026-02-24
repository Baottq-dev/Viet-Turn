#!/usr/bin/env python3
"""
00b_split_audio.py - Split long audio into ~10 min segments using Silero VAD

Splits long conversation audio (30-90 min) into shorter segments (~10 min)
suitable for diarization and VAP training. Uses Silero VAD (neural network)
to detect speech pauses, then cuts at the longest pause near each target
split point — ensuring no utterance is cut mid-speech.

Algorithm:
    1. Run Silero VAD to get speech timestamps
    2. Invert speech regions to find pause/silence gaps
    3. For each target split point (~10 min), pick the longest pause
       within a ±tolerance window
    4. Cut at the midpoint of that pause

Usage:
    python scripts/00b_split_audio.py --input data/audio --output data/audio_split
    python scripts/00b_split_audio.py --input data/audio --output data/audio_split --segment-min 10
    python scripts/00b_split_audio.py --input data/audio --output data/audio_split --dry-run
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.audio import get_audio_duration


SAMPLE_RATE = 16000


def load_silero_vad():
    """Load Silero VAD model from torch.hub (cached after first download)."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def find_pauses_from_vad(
    waveform: torch.Tensor,
    sample_rate: int,
    vad_model,
    get_speech_ts,
    min_pause_sec: float = 0.3,
) -> List[Tuple[float, float, float]]:
    """
    Run Silero VAD and return pause regions (gaps between speech).

    Args:
        waveform: (num_samples,) audio tensor, 16kHz.
        sample_rate: Audio sample rate.
        vad_model: Silero VAD model.
        get_speech_ts: get_speech_timestamps function from Silero utils.
        min_pause_sec: Minimum pause duration to keep (seconds).

    Returns:
        List of (start_sec, end_sec, duration_sec) for each pause.
    """
    speech_timestamps = get_speech_ts(
        waveform,
        vad_model,
        sampling_rate=sample_rate,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        speech_pad_ms=30,
    )

    if not speech_timestamps:
        return []

    total_samples = len(waveform)
    pauses = []

    # Pause before first speech
    first_start = speech_timestamps[0]["start"]
    if first_start > 0:
        dur = first_start / sample_rate
        if dur >= min_pause_sec:
            pauses.append((0.0, first_start / sample_rate, dur))

    # Pauses between speech segments
    for i in range(len(speech_timestamps) - 1):
        gap_start = speech_timestamps[i]["end"]
        gap_end = speech_timestamps[i + 1]["start"]
        dur = (gap_end - gap_start) / sample_rate
        if dur >= min_pause_sec:
            pauses.append((
                gap_start / sample_rate,
                gap_end / sample_rate,
                dur,
            ))

    # Pause after last speech
    last_end = speech_timestamps[-1]["end"]
    if last_end < total_samples:
        dur = (total_samples - last_end) / sample_rate
        if dur >= min_pause_sec:
            pauses.append((last_end / sample_rate, total_samples / sample_rate, dur))

    return pauses


def find_best_split_point(
    pauses: List[Tuple[float, float, float]],
    target_sec: float,
    tolerance_sec: float = 60.0,
) -> float:
    """
    Find the best split point near a target time.

    Picks the longest pause within [target - tolerance, target + tolerance],
    preferring pauses closer to target as tiebreaker.

    Args:
        pauses: List of (start_sec, end_sec, duration_sec).
        target_sec: Desired split time in seconds.
        tolerance_sec: Search window around target.

    Returns:
        Split time in seconds (midpoint of chosen pause).
    """
    candidates = []
    for start, end, dur in pauses:
        mid = (start + end) / 2
        if target_sec - tolerance_sec <= mid <= target_sec + tolerance_sec:
            distance = abs(mid - target_sec)
            candidates.append((dur, -distance, mid))

    if not candidates:
        return target_sec

    # Sort by: longest pause first, then closest to target
    candidates.sort(reverse=True)
    return candidates[0][2]


def split_audio_file(
    input_path: Path,
    output_dir: Path,
    vad_model,
    get_speech_ts,
    segment_min: float = 10.0,
    tolerance_sec: float = 60.0,
    min_segment_sec: float = 120.0,
    min_pause_sec: float = 0.3,
    dry_run: bool = False,
) -> List[dict]:
    """
    Split a single audio file into segments at VAD-detected pauses.

    Args:
        input_path: Path to input WAV file.
        output_dir: Output directory for segments.
        vad_model: Silero VAD model.
        get_speech_ts: get_speech_timestamps function.
        segment_min: Target segment duration in minutes.
        tolerance_sec: Search window around target split points.
        min_segment_sec: Don't split files shorter than this (seconds).
        min_pause_sec: Minimum pause duration for split candidates.
        dry_run: If True, only report split points without writing files.

    Returns:
        List of segment info dicts.
    """
    duration = get_audio_duration(input_path)
    file_id = input_path.stem
    segment_sec = segment_min * 60

    # Skip short files
    if duration < min_segment_sec:
        print(f"  [SKIP] {file_id}: {duration:.0f}s < {min_segment_sec:.0f}s minimum")
        return []

    # Don't split if already close to target
    if duration <= segment_sec + tolerance_sec:
        print(f"  [KEEP] {file_id}: {duration:.0f}s — already near target, no split needed")
        return [{"file_id": file_id, "start": 0, "end": duration, "duration": duration, "path": str(input_path)}]

    print(f"  Loading {file_id} ({duration:.0f}s = {duration/60:.1f} min)...")

    # Load full audio
    waveform, sr = torchaudio.load(str(input_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform.unsqueeze(0)).squeeze(0)

    # Run VAD to find pauses
    print(f"  Running Silero VAD...")
    pauses = find_pauses_from_vad(
        waveform, SAMPLE_RATE, vad_model, get_speech_ts,
        min_pause_sec=min_pause_sec,
    )
    print(f"  Found {len(pauses)} pauses (>= {min_pause_sec}s)")

    # Determine split points
    num_segments = max(1, round(duration / segment_sec))
    split_points = [0.0]

    for i in range(1, num_segments):
        target = i * segment_sec
        best = find_best_split_point(pauses, target, tolerance_sec)
        # Ensure minimum gap from previous split
        if best - split_points[-1] < 60:
            best = target  # fallback
        split_points.append(best)

    split_points.append(duration)

    # Report split plan
    segments = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        seg_dur = end - start
        seg_id = f"{file_id}_{i:03d}"
        seg_path = output_dir / f"{seg_id}.wav"

        segments.append({
            "file_id": seg_id,
            "source_file": file_id,
            "segment_idx": i,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(seg_dur, 3),
            "path": str(seg_path),
        })

        start_fmt = f"{int(start//60)}:{int(start%60):02d}"
        end_fmt = f"{int(end//60)}:{int(end%60):02d}"
        print(f"    [{i}] {seg_id}: {start_fmt} -> {end_fmt} ({seg_dur:.0f}s)")

    if dry_run:
        return segments

    # Write segment files
    output_dir.mkdir(parents=True, exist_ok=True)

    for seg in segments:
        start_sample = int(seg["start"] * SAMPLE_RATE)
        end_sample = int(seg["end"] * SAMPLE_RATE)
        segment_audio = waveform[start_sample:end_sample].unsqueeze(0)

        seg_path = Path(seg["path"])
        torchaudio.save(str(seg_path), segment_audio, SAMPLE_RATE)

    print(f"  Wrote {len(segments)} segments to {output_dir}")
    return segments


def format_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="Split long audio into ~10 min segments using Silero VAD"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input directory with WAV files (or single file)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for split segments",
    )
    parser.add_argument(
        "--segment-min", type=float, default=10.0,
        help="Target segment duration in minutes (default: 10)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=60.0,
        help="Search window in seconds around target split point (default: 60)",
    )
    parser.add_argument(
        "--min-duration", type=float, default=120.0,
        help="Skip files shorter than this many seconds (default: 120)",
    )
    parser.add_argument(
        "--min-pause", type=float, default=0.3,
        help="Minimum pause duration in seconds for split candidates (default: 0.3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show split plan without writing files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Collect input files
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.wav"))

    if not files:
        print(f"[ERROR] No WAV files found in {input_path}")
        sys.exit(1)

    # Load VAD model once
    print("Loading Silero VAD model...")
    vad_model, get_speech_ts = load_silero_vad()
    print()

    print(f"=== Split Audio into ~{args.segment_min:.0f} min Segments (Silero VAD) ===\n")
    print(f"Input:      {input_path}")
    print(f"Output:     {output_dir}")
    print(f"Files:      {len(files)}")
    print(f"Target:     ~{args.segment_min:.0f} min/segment")
    print(f"Tolerance:  +/-{args.tolerance:.0f}s around target")
    print(f"Min pause:  {args.min_pause}s")
    if args.dry_run:
        print(f"Mode:       DRY RUN")
    print()

    all_segments = []

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}")
        segments = split_audio_file(
            f, output_dir,
            vad_model, get_speech_ts,
            segment_min=args.segment_min,
            tolerance_sec=args.tolerance,
            min_segment_sec=args.min_duration,
            min_pause_sec=args.min_pause,
            dry_run=args.dry_run,
        )
        all_segments.extend(segments)
        print()

    # Summary
    total_dur = sum(s["duration"] for s in all_segments)
    avg_dur = total_dur / len(all_segments) if all_segments else 0

    print(f"=== Summary ===")
    print(f"Input files:     {len(files)}")
    print(f"Output segments: {len(all_segments)}")
    print(f"Total duration:  {format_duration(total_dur)}")
    print(f"Avg segment:     {format_duration(avg_dur)}")

    if all_segments and not args.dry_run:
        print(f"\nNext step: python scripts/01_diarize.py --input {output_dir} --output data/rttm")


if __name__ == "__main__":
    main()
