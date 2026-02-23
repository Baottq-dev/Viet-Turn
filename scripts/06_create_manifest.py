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


def split_by_file(
    entries: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split entries into train/val/test by file (conversation-level split).
    """
    rng = random.Random(seed)
    file_ids = [e["file_id"] for e in entries]
    rng.shuffle(file_ids)

    n = len(file_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(file_ids[:n_train])
    val_ids = set(file_ids[n_train:n_train + n_val])
    test_ids = set(file_ids[n_train + n_val:])

    train = [e for e in entries if e["file_id"] in train_ids]
    val = [e for e in entries if e["file_id"] in val_ids]
    test = [e for e in entries if e["file_id"] in test_ids]

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
    split_info = {
        "total_files": len(entries),
        "train_files": len(train),
        "val_files": len(val),
        "test_files": len(test),
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
