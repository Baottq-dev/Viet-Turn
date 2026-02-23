#!/usr/bin/env python3
"""
05_align_text.py - Align Text to Frames

Maps word-level timestamps from ASR transcripts to 50fps frame indices.
Produces cumulative text at each frame for PhoBERT input during training.

Usage:
    python scripts/05_align_text.py \
        --transcripts data/transcripts \
        --va-matrices data/va_matrices \
        --output data/text_frames
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.audio import seconds_to_frames

FRAME_HZ = 50


def align_words_to_frames(
    words: List[Dict],
    num_frames: int,
    frame_hz: int = FRAME_HZ,
) -> Dict:
    """
    Align word timestamps to frame indices.

    For each frame, determine what cumulative text is available up to that point.
    This simulates streaming ASR: at frame t, only words that ended before t are available.

    Args:
        words: [{word, start, end}, ...] sorted by start time.
        num_frames: Total number of frames in the audio.
        frame_hz: Frame rate.

    Returns:
        alignment: {
            "word_end_frames": [frame_idx, ...] for each word,
            "frame_to_word_idx": [last_completed_word_idx, ...] for each frame (-1 if none),
            "num_words": int,
            "num_frames": int,
        }
    """
    # Compute end frame for each word
    word_end_frames = []
    for w in words:
        end_frame = seconds_to_frames(w["end"], frame_hz)
        word_end_frames.append(min(end_frame, num_frames - 1))

    # For each frame, find the index of the last word that completed before/at this frame
    frame_to_word_idx = np.full(num_frames, -1, dtype=np.int32)
    word_idx = 0
    for t in range(num_frames):
        while word_idx < len(words) and word_end_frames[word_idx] <= t:
            word_idx += 1
        frame_to_word_idx[t] = word_idx - 1  # -1 means no word completed yet

    return {
        "word_end_frames": word_end_frames,
        "frame_to_word_idx": frame_to_word_idx.tolist(),
        "num_words": len(words),
        "num_frames": num_frames,
    }


def build_text_windows(
    words: List[Dict],
    alignment: Dict,
    max_words_context: int = 128,
) -> List[Dict]:
    """
    Build text snapshots at key frames (when new words arrive).

    Instead of storing text for every frame (wasteful), store text only at
    word boundaries. During training, the dataset will look up the latest
    text snapshot for each frame.

    Returns:
        text_snapshots: [{frame, word_idx, text}, ...]
        First entry is always {frame: 0, word_idx: -1, text: ""}.
    """
    snapshots = [{"frame": 0, "word_idx": -1, "text": ""}]

    for w_idx, w in enumerate(words):
        end_frame = alignment["word_end_frames"][w_idx]
        # Cumulative text (last max_words_context words)
        start_idx = max(0, w_idx + 1 - max_words_context)
        text = " ".join(words[j]["word"] for j in range(start_idx, w_idx + 1))
        snapshots.append({
            "frame": end_frame,
            "word_idx": w_idx,
            "text": text,
        })

    return snapshots


def main():
    parser = argparse.ArgumentParser(description="Align text to frames")
    parser.add_argument("--transcripts", required=True, help="Directory with transcript JSONs")
    parser.add_argument("--va-matrices", required=True, help="Directory with VA matrix .pt files (for frame count)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max-words", type=int, default=128, help="Max words in context window")
    args = parser.parse_args()

    transcript_dir = Path(args.transcripts)
    va_dir = Path(args.va_matrices)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    transcript_files = sorted(transcript_dir.glob("*.json"))
    # Skip summary files
    transcript_files = [f for f in transcript_files if not f.name.startswith("_")]
    print(f"Found {len(transcript_files)} transcript files")

    results = {}
    for i, transcript_path in enumerate(transcript_files):
        file_id = transcript_path.stem
        print(f"\n[{i+1}/{len(transcript_files)}] Aligning {file_id}...")

        try:
            # Load transcript
            with open(transcript_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            words = data["words"]

            # Get num_frames from VA matrix
            va_path = va_dir / f"{file_id}.pt"
            if not va_path.exists():
                print(f"  WARNING: VA matrix not found for {file_id}, skipping")
                results[file_id] = {"status": "error", "error": "va_matrix not found"}
                continue

            va_matrix = torch.load(va_path, weights_only=True)
            num_frames = va_matrix.shape[1]

            # Align
            alignment = align_words_to_frames(words, num_frames)

            # Build text snapshots
            snapshots = build_text_windows(words, alignment, args.max_words)

            # Save
            output_data = {
                "file_id": file_id,
                "num_frames": num_frames,
                "num_words": len(words),
                "num_snapshots": len(snapshots),
                "words": words,
                "alignment": {
                    "word_end_frames": alignment["word_end_frames"],
                },
                "text_snapshots": snapshots,
            }

            output_path = output_dir / f"{file_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            print(f"  {len(words)} words, {num_frames} frames, {len(snapshots)} text snapshots")

            results[file_id] = {
                "status": "ok",
                "num_words": len(words),
                "num_frames": num_frames,
                "num_snapshots": len(snapshots),
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            results[file_id] = {"status": "error", "error": str(e)}

    # Summary
    summary_path = output_dir / "_alignment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"\nDone: {ok}/{len(results)} files aligned")


if __name__ == "__main__":
    main()
