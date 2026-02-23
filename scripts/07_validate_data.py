#!/usr/bin/env python3
"""
07_validate_data.py - Validate Dataset Quality

Checks the quality of the generated dataset:
- Label distribution (expect ~70% silence class)
- Frame count consistency across modalities
- Flags problematic files (too much silence, too much overlap)

Usage:
    python scripts/07_validate_data.py --manifest data/vap_manifest_train.json
    python scripts/07_validate_data.py --va-dir data/va_matrices --label-dir data/vap_labels
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict
from collections import Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.labels import get_label_statistics, decode_vap_labels, NUM_CLASSES


def validate_file(
    file_id: str,
    va_matrix_path: Path,
    vap_label_path: Path,
    text_frames_path: Path = None,
) -> Dict:
    """Validate a single file's data."""
    issues = []

    # Load VA matrix
    va_matrix = torch.load(va_matrix_path, weights_only=True).numpy()
    num_frames = va_matrix.shape[1]

    # Load labels
    labels = torch.load(vap_label_path, weights_only=True).numpy()

    # Check 1: Frame count consistency
    if labels.shape[0] != num_frames:
        issues.append(f"Frame mismatch: VA={num_frames}, labels={labels.shape[0]}")

    # Check 2: Label range
    valid_labels = labels[labels >= 0]
    if len(valid_labels) > 0:
        if valid_labels.max() >= NUM_CLASSES:
            issues.append(f"Label out of range: max={valid_labels.max()}")
        if valid_labels.min() < 0:
            issues.append(f"Negative valid label: min={valid_labels.min()}")

    # Check 3: VA matrix values
    if not np.all((va_matrix == 0) | (va_matrix == 1)):
        issues.append("VA matrix contains non-binary values")

    # Check 4: Silence ratio
    silence = ((va_matrix[0] == 0) & (va_matrix[1] == 0)).mean()
    if silence > 0.8:
        issues.append(f"Excessive silence: {silence:.1%}")

    # Check 5: Overlap ratio
    overlap = ((va_matrix[0] > 0) & (va_matrix[1] > 0)).mean()
    if overlap > 0.5:
        issues.append(f"Excessive overlap: {overlap:.1%}")

    # Check 6: Single speaker dominance
    s0_ratio = va_matrix[0].mean()
    s1_ratio = va_matrix[1].mean()
    if s0_ratio > 0 and s1_ratio > 0:
        dominance = max(s0_ratio, s1_ratio) / (s0_ratio + s1_ratio)
        if dominance > 0.9:
            issues.append(f"Speaker dominance: {dominance:.1%}")

    # Check 7: Text alignment (if available)
    if text_frames_path and text_frames_path.exists():
        with open(text_frames_path, "r", encoding="utf-8") as f:
            text_data = json.load(f)
        if text_data.get("num_frames") != num_frames:
            issues.append(f"Text frame mismatch: text={text_data.get('num_frames')}, VA={num_frames}")

    # Label stats
    stats = get_label_statistics(labels)

    return {
        "file_id": file_id,
        "num_frames": num_frames,
        "valid_frames": int((labels >= 0).sum()),
        "silence_ratio": round(float(silence), 4),
        "overlap_ratio": round(float(overlap), 4),
        "speaker_0_ratio": round(float(s0_ratio), 4),
        "speaker_1_ratio": round(float(s1_ratio), 4),
        "unique_classes": stats.get("unique_classes", 0),
        "issues": issues,
        "status": "warning" if issues else "ok",
    }



def main():
    parser = argparse.ArgumentParser(description="Validate dataset quality")
    parser.add_argument("--manifest", default=None, help="Manifest JSON file")
    parser.add_argument("--va-dir", default=None, help="VA matrix directory")
    parser.add_argument("--label-dir", default=None, help="VAP label directory")
    parser.add_argument("--text-dir", default=None, help="Text frames directory")
    args = parser.parse_args()

    # Gather files
    files = []
    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for e in entries:
            files.append({
                "file_id": e["file_id"],
                "va_path": Path(e["va_matrix_path"]),
                "label_path": Path(e["vap_label_path"]),
                "text_path": Path(e.get("text_frames_path", "")),
            })
    elif args.va_dir and args.label_dir:
        va_dir = Path(args.va_dir)
        label_dir = Path(args.label_dir)
        text_dir = Path(args.text_dir) if args.text_dir else None

        for va_path in sorted(va_dir.glob("*.pt")):
            file_id = va_path.stem
            label_path = label_dir / f"{file_id}.pt"
            text_path = (text_dir / f"{file_id}.json") if text_dir else None

            if label_path.exists():
                files.append({
                    "file_id": file_id,
                    "va_path": va_path,
                    "label_path": label_path,
                    "text_path": text_path,
                })
    else:
        print("ERROR: Provide --manifest or --va-dir + --label-dir")
        sys.exit(1)

    print(f"Validating {len(files)} files...\n")

    all_results = []
    all_labels = []
    warnings = 0

    for i, f in enumerate(files):
        result = validate_file(
            f["file_id"],
            f["va_path"],
            f["label_path"],
            f.get("text_path"),
        )
        all_results.append(result)

        if result["issues"]:
            warnings += 1
            print(f"[{i+1}] {f['file_id']}: {', '.join(result['issues'])}")

        # Collect labels for global stats
        labels = torch.load(f["label_path"], weights_only=True).numpy()
        valid = labels[labels >= 0]
        all_labels.extend(valid.tolist())

    # Global statistics
    print(f"\n{'='*60}")
    print(f"Total files: {len(files)}")
    print(f"Files with warnings: {warnings}")
    print(f"Files OK: {len(files) - warnings}")

    if all_labels:
        all_labels = np.array(all_labels)
        print(f"\nGlobal label statistics:")
        print(f"  Total valid frames: {len(all_labels):,}")
        print(f"  Unique classes: {len(np.unique(all_labels))}/256")

        # Class distribution
        counter = Counter(all_labels.tolist())
        most_common = counter.most_common(10)
        print(f"  Top 10 classes:")
        for cls, count in most_common:
            decoded = decode_vap_labels(np.array([cls]))[0]
            s0_bins = "".join(str(b) for b in decoded[0])
            s1_bins = "".join(str(b) for b in decoded[1])
            print(f"    Class {cls:3d} (S0={s0_bins} S1={s1_bins}): {count:>8,} ({count/len(all_labels):.1%})")

        # Check class 0 (all silent) ratio
        silence_class_ratio = counter.get(0, 0) / len(all_labels)
        print(f"\n  Silence class (0) ratio: {silence_class_ratio:.1%}")
        if silence_class_ratio > 0.8:
            print("  WARNING: Dataset is dominated by silence. Consider filtering.")

    # Save report
    output_path = Path(args.manifest).parent if args.manifest else Path(args.va_dir).parent
    report_path = output_path / "_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_files": len(files),
            "warnings": warnings,
            "total_valid_frames": len(all_labels) if all_labels is not None else 0,
            "files": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
