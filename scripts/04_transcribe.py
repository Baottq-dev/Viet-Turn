#!/usr/bin/env python3
"""
04_transcribe.py - ASR Transcription with Word Timestamps

Runs faster-whisper on audio files to produce word-level transcripts
with timestamps for text-frame alignment.

Usage:
    python scripts/04_transcribe.py --input data/audio --output data/transcripts
    python scripts/04_transcribe.py --input data/audio --output data/transcripts --model large-v3
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from faster_whisper import WhisperModel


def get_audio_files(input_path: Path) -> List[Path]:
    """Get all audio files from input path."""
    if input_path.is_file():
        return [input_path]
    extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    return sorted(f for f in input_path.rglob("*") if f.suffix.lower() in extensions)


def transcribe_file(
    audio_path: Path,
    model: WhisperModel,
    language: str = "vi",
    beam_size: int = 5,
) -> Tuple[List[Dict], str]:
    """
    Transcribe a single audio file with word-level timestamps.

    Returns:
        word_segments: [{word, start, end}, ...]
        full_transcript: Concatenated text.
    """
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        beam_size=beam_size,
        vad_filter=True,
    )

    word_segments = []
    for seg in segments_iter:
        for w in (seg.words or []):
            word_segments.append({
                "word": w.word.strip(),
                "start": round(w.start, 3),
                "end": round(w.end, 3),
            })

    full_transcript = " ".join(w["word"] for w in word_segments)
    return word_segments, full_transcript


def main():
    parser = argparse.ArgumentParser(description="ASR transcription with word timestamps")
    parser.add_argument("--input", required=True, help="Input audio file or directory")
    parser.add_argument("--output", required=True, help="Output directory for transcripts")
    parser.add_argument("--model", default="large-v3", help="Whisper model name")
    parser.add_argument("--language", default="vi", help="Language code")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device == "cuda" else "int8"

    # Load model
    print(f"Loading faster-whisper ({args.model}) on {device}...")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    # Process files
    audio_files = get_audio_files(input_path)
    print(f"Found {len(audio_files)} audio files")

    results = {}
    for i, audio_path in enumerate(audio_files):
        file_id = audio_path.stem
        output_path = output_dir / f"{file_id}.json"

        # Skip if already transcribed (resume support)
        if output_path.exists():
            print(f"\n[{i+1}/{len(audio_files)}] {file_id} — already done, skipping")
            continue

        print(f"\n[{i+1}/{len(audio_files)}] Transcribing {file_id}...")

        try:
            word_segments, full_transcript = transcribe_file(
                audio_path, model,
                language=args.language,
                beam_size=args.beam_size,
            )

            # Save transcript
            transcript_data = {
                "file_id": file_id,
                "num_words": len(word_segments),
                "transcript": full_transcript,
                "words": word_segments,
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)

            duration = word_segments[-1]["end"] if word_segments else 0
            print(f"  {len(word_segments)} words, {duration:.1f}s")

            results[file_id] = {
                "status": "ok",
                "num_words": len(word_segments),
                "duration_sec": round(duration, 1),
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            results[file_id] = {"status": "error", "error": str(e)}

    # Summary
    summary_path = output_dir / "_transcription_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"\nDone: {ok}/{len(results)} files transcribed")


if __name__ == "__main__":
    main()
