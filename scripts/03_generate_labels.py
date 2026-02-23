#!/usr/bin/env python3
"""
03_generate_labels.py - Generate VAP Labels from Voice Activity Matrices

Converts binary VA matrices (2, num_frames) into 256-class VAP labels
using the self-supervised encoding scheme.

Usage:
    python scripts/03_generate_labels.py --input data/va_matrices --output data/vap_labels
"""

import sys
import json
import argparse
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.labels import encode_vap_labels, get_label_statistics


def main():
    parser = argparse.ArgumentParser(description="Generate VAP labels from VA matrices")
    parser.add_argument("--input", required=True, help="Directory with VA matrix .pt files")
    parser.add_argument("--output", required=True, help="Output directory for VAP labels")
    parser.add_argument("--activity-threshold", type=float, default=0.5,
                        help="Fraction of bin that must be active")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    va_files = sorted(input_dir.glob("*.pt"))
    print(f"Found {len(va_files)} VA matrix files")

    all_stats = {}
    total_valid = 0
    total_invalid = 0

    for i, va_path in enumerate(va_files):
        file_id = va_path.stem
        print(f"\n[{i+1}/{len(va_files)}] Processing {file_id}...")

        try:
            # Load VA matrix
            va_matrix = torch.load(va_path, weights_only=True).numpy()
            assert va_matrix.shape[0] == 2, f"Expected 2 speakers, got {va_matrix.shape[0]}"

            # Generate labels
            labels = encode_vap_labels(va_matrix, activity_threshold=args.activity_threshold)

            # Save
            output_path = output_dir / f"{file_id}.pt"
            torch.save(torch.from_numpy(labels), output_path)

            # Stats
            stats = get_label_statistics(labels)
            valid = stats["valid_frames"]
            invalid = stats["invalid_frames"]
            total_valid += valid
            total_invalid += invalid

            print(f"  {stats['total_frames']} frames total, {valid} valid, {invalid} invalid (tail)")
            print(f"  {stats['unique_classes']} unique classes out of 256")
            if stats.get("top_classes"):
                top3 = stats["top_classes"][:3]
                top_str = ", ".join(f"class {c['class']}={c['ratio']:.1%}" for c in top3)
                print(f"  Top classes: {top_str}")

            all_stats[file_id] = stats

        except Exception as e:
            print(f"  ERROR: {e}")
            all_stats[file_id] = {"status": "error", "error": str(e)}

    # Global summary
    print(f"\n{'='*60}")
    print(f"Total: {total_valid} valid frames, {total_invalid} invalid frames")
    print(f"Valid ratio: {total_valid / max(1, total_valid + total_invalid):.1%}")

    summary_path = output_dir / "_label_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
