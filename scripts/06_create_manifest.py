#!/usr/bin/env python3
"""
06_create_manifest.py - Create Dataset Manifest

Combines all pipeline outputs (audio, VA matrices, VAP labels, text alignments)
into train/val/test manifest files. Splits by conversation (not by window)
to prevent data leakage.

Usage:
    python scripts/06_create_manifest.py \
        --audio-dir data/audio \
        --va-dir data/va_matrices \
        --label-dir data/vap_labels \
        --text-dir data/text_frames \
        --output data
"""

import sys
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


def gather_files(
    audio_dir: Path,
    va_dir: Path,
    label_dir: Path,
    text_dir: Path,
) -> List[Dict]:
    """
    Gather all complete entries (files that have all 4 components).
    """
    # Find all audio files
    audio_files = {}
    for ext in [".wav", ".mp3", ".flac", ".m4a"]:
        for f in audio_dir.glob(f"*{ext}"):
            audio_files[f.stem] = f

    entries = []
    for file_id, audio_path in sorted(audio_files.items()):
        va_path = va_dir / f"{file_id}.pt"
        label_path = label_dir / f"{file_id}.pt"
        text_path = text_dir / f"{file_id}.json"

        if not va_path.exists():
            print(f"  SKIP {file_id}: missing VA matrix")
            continue
        if not label_path.exists():
            print(f"  SKIP {file_id}: missing VAP labels")
            continue
        if not text_path.exists():
            print(f"  SKIP {file_id}: missing text alignment")
            continue

        entries.append({
            "file_id": file_id,
            "audio_path": str(audio_path),
            "va_matrix_path": str(va_path),
            "vap_label_path": str(label_path),
            "text_frames_path": str(text_path),
        })

    return entries


def _get_source_video(file_id: str) -> str:
    """Extract source video ID from segment file_id.

    '92tmp9tXzso_003' -> '92tmp9tXzso'
    'NdGFee__zGc_006' -> 'NdGFee__zGc'  (handles double underscore in video ID)
    """
    # Segment suffix is always _NNN (3 digits) at the end
    if len(file_id) >= 4 and file_id[-4] == "_" and file_id[-3:].isdigit():
        return file_id[:-4]
    return file_id


def split_by_file(
    entries: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split entries into train/val/test by SOURCE VIDEO to prevent data leakage.

    All segments from the same source video go into the same split,
    so the model never sees the same speakers in both train and test.
    """
    rng = random.Random(seed)

    # Group segments by source video
    video_to_entries = defaultdict(list)
    for e in entries:
        source = _get_source_video(e["file_id"])
        video_to_entries[source].append(e)

    # Shuffle source videos (not individual segments)
    source_videos = sorted(video_to_entries.keys())
    rng.shuffle(source_videos)

    n = len(source_videos)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_videos = set(source_videos[:n_train])
    val_videos = set(source_videos[n_train:n_train + n_val])
    test_videos = set(source_videos[n_train + n_val:])

    train = [e for src in train_videos for e in video_to_entries[src]]
    val = [e for src in val_videos for e in video_to_entries[src]]
    test = [e for src in test_videos for e in video_to_entries[src]]

    # Sort within each split for deterministic order
    train.sort(key=lambda e: e["file_id"])
    val.sort(key=lambda e: e["file_id"])
    test.sort(key=lambda e: e["file_id"])

    print(f"\n  Source videos: {n} total -> "
          f"train={len(train_videos)}, val={len(val_videos)}, test={len(test_videos)}")

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Create dataset manifest")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--va-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--output", required=True, help="Output directory for manifest files")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Gathering files...")
    entries = gather_files(
        Path(args.audio_dir),
        Path(args.va_dir),
        Path(args.label_dir),
        Path(args.text_dir),
    )
    print(f"Found {len(entries)} complete entries")

    if len(entries) == 0:
        print("ERROR: No complete entries found. Check that all pipeline steps have run.")
        sys.exit(1)

    # Split
    train, val, test = split_by_file(
        entries,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"\nSplit: train={len(train)}, val={len(val)}, test={len(test)}")

    # Save manifests
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        manifest_path = output_dir / f"vap_manifest_{split_name}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  {split_name}: {len(split_data)} files -> {manifest_path}")

    # Save split info
    train_sources = sorted(set(_get_source_video(e["file_id"]) for e in train))
    val_sources = sorted(set(_get_source_video(e["file_id"]) for e in val))
    test_sources = sorted(set(_get_source_video(e["file_id"]) for e in test))

    split_info = {
        "total_files": len(entries),
        "total_source_videos": len(train_sources) + len(val_sources) + len(test_sources),
        "train_files": len(train),
        "val_files": len(val),
        "test_files": len(test),
        "train_source_videos": train_sources,
        "val_source_videos": val_sources,
        "test_source_videos": test_sources,
        "train_ids": [e["file_id"] for e in train],
        "val_ids": [e["file_id"] for e in val],
        "test_ids": [e["file_id"] for e in test],
        "seed": args.seed,
    }
    with open(output_dir / "vap_split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
