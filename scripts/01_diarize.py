#!/usr/bin/env python3
"""
01_diarize.py - Speaker Diarization with pyannote

Runs pyannote/speaker-diarization-3.1 on mono audio files to produce
RTTM files with per-speaker voice activity segments.

Usage:
    python scripts/01_diarize.py --input data/audio --output data/rttm
    python scripts/01_diarize.py --input data/audio/file.wav --output data/rttm
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from pyannote.audio import Pipeline


def get_audio_files(input_path: Path) -> List[Path]:
    """Get all audio files from input path."""
    if input_path.is_file():
        return [input_path]
    extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    files = [f for f in input_path.rglob("*") if f.suffix.lower() in extensions]
    return sorted(files)


def diarize_file(
    audio_path: Path,
    pipeline: Pipeline,
    num_speakers: int = 2,
    min_duration: float = 0.1,
) -> List[Dict]:
    """
    Run diarization on a single audio file.

    Args:
        audio_path: Path to audio file.
        pipeline: pyannote diarization pipeline.
        num_speakers: Expected number of speakers.
        min_duration: Minimum segment duration in seconds.

    Returns:
        List of segments: [{speaker, start, end, duration}, ...]
    """
    diarization = pipeline(
        str(audio_path),
        num_speakers=num_speakers,
    )

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = turn.end - turn.start
        if duration >= min_duration:
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "duration": round(duration, 3),
            })

    return segments


def write_rttm(segments: List[Dict], output_path: Path, file_id: str):
    """Write segments to RTTM format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            duration = seg["end"] - seg["start"]
            f.write(
                f"SPEAKER {file_id} 1 {seg['start']:.3f} {duration:.3f} "
                f"<NA> <NA> {seg['speaker']} <NA> <NA>\n"
            )


def write_json(segments: List[Dict], output_path: Path):
    """Write segments to JSON format (for easier processing)."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Speaker diarization with pyannote")
    parser.add_argument("--input", required=True, help="Input audio file or directory")
    parser.add_argument("--output", required=True, help="Output directory for RTTM files")
    parser.add_argument("--num-speakers", type=int, default=2, help="Expected number of speakers")
    parser.add_argument("--min-duration", type=float, default=0.1, help="Minimum segment duration (sec)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for pyannote")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pipeline
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    print(f"Loading pyannote/speaker-diarization-3.1 on {device}...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline.to(device)

    # Process files
    audio_files = get_audio_files(input_path)
    print(f"Found {len(audio_files)} audio files")

    results = {}
    for i, audio_path in enumerate(audio_files):
        file_id = audio_path.stem
        print(f"\n[{i+1}/{len(audio_files)}] Processing {file_id}...")

        try:
            segments = diarize_file(
                audio_path, pipeline,
                num_speakers=args.num_speakers,
                min_duration=args.min_duration,
            )

            # Write RTTM
            rttm_path = output_dir / f"{file_id}.rttm"
            write_rttm(segments, rttm_path, file_id)

            # Write JSON (easier to process later)
            json_path = output_dir / f"{file_id}.json"
            write_json(segments, json_path)

            # Stats
            speakers = set(s["speaker"] for s in segments)
            total_dur = sum(s["duration"] for s in segments)
            print(f"  {len(segments)} segments, {len(speakers)} speakers, {total_dur:.1f}s total speech")

            results[file_id] = {
                "status": "ok",
                "num_segments": len(segments),
                "num_speakers": len(speakers),
                "total_speech_sec": round(total_dur, 1),
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            results[file_id] = {"status": "error", "error": str(e)}

    # Summary
    summary_path = output_dir / "_diarization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"\nDone: {ok}/{len(results)} files processed successfully")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
