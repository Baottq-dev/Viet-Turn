#!/usr/bin/env python3
"""
05_prepare_dataset.py - Convert LLM-processed data to training-ready format

Chuyển đổi dữ liệu từ llm_processed/ (output của 02_llm_process.py)
sang format chuẩn cho 06_split_dataset.py:
  - Parse timestamps MM:SS → float seconds
  - Merge 5-class labels → 3-class (YIELD, HOLD, BACKCHANNEL)
  - Validate segments và fix edge cases
  - Output JSON files với format "segments[]"

Usage:
    python scripts/05_prepare_dataset.py
    python scripts/05_prepare_dataset.py --input datasets/processed/llm_processed --output datasets/processed/final
    python scripts/05_prepare_dataset.py --dry-run
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Encoding fix for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# CONSTANTS
# ==============================================================================

LABEL_MERGE_MAP = {
    "YIELD": "YIELD",
    "HOLD": "HOLD",
    "BACKCHANNEL": "BACKCHANNEL",
    "COOPERATIVE_INTERRUPT": "BACKCHANNEL",
    "COMPETITIVE_INTERRUPT": "YIELD",
}

VALID_LABELS = {"YIELD", "HOLD", "BACKCHANNEL"}


# ==============================================================================
# CONVERSION FUNCTIONS
# ==============================================================================

def parse_timestamp(ts: str) -> float:
    """Convert timestamp string to float seconds.

    Supports:
        "MM:SS"    → minutes * 60 + seconds
        "HH:MM:SS" → hours * 3600 + minutes * 60 + seconds
        "SS"       → seconds (fallback)

    Examples:
        "00:00" → 0.0
        "09:12" → 552.0
        "1:30:00" → 5400.0
    """
    ts = ts.strip()
    parts = ts.split(":")

    if len(parts) == 2:
        minutes, seconds = int(parts[0]), int(parts[1])
        return float(minutes * 60 + seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        return float(hours * 3600 + minutes * 60 + seconds)
    elif len(parts) == 1:
        return float(parts[0])
    else:
        raise ValueError(f"Unrecognized timestamp format: '{ts}'")


def merge_label(label: str) -> str:
    """Map 5-class label to 3-class label.

    COOPERATIVE_INTERRUPT → BACKCHANNEL
    COMPETITIVE_INTERRUPT → YIELD
    """
    merged = LABEL_MERGE_MAP.get(label)
    if merged is None:
        return "YIELD"
    return merged


def convert_turn_to_segment(turn: Dict) -> Dict:
    """Convert a single LLM turn dict to target segment dict."""
    original_label = turn.get("turn_taking_label", "YIELD")
    merged = merge_label(original_label)

    segment = {
        "id": turn.get("turn_id", 0),
        "start": parse_timestamp(turn.get("start_time", "0:00")),
        "end": parse_timestamp(turn.get("end_time", "0:00")),
        "text": turn.get("text", ""),
        "speaker": turn.get("speaker", "SPEAKER_01"),
        "label": merged,
        "reviewed": True,
        "confidence": turn.get("confidence", 0.8),
        "label_reason": turn.get("label_reason", ""),
    }

    # Preserve original label if it was merged (for traceability)
    if original_label != merged:
        segment["original_label"] = original_label

    return segment


def validate_and_fix_segment(
    segment: Dict, audio_file: str, min_duration: float
) -> Tuple[Dict, List[str]]:
    """Validate a segment and fix common issues.

    Returns:
        (fixed_segment, list_of_warnings)
    """
    warnings = []
    seg_id = segment.get("id", "?")
    prefix = f"  [{audio_file}] segment {seg_id}"

    # Fix zero or negative duration
    if segment["start"] >= segment["end"]:
        old_end = segment["end"]
        segment["end"] = segment["start"] + min_duration
        warnings.append(
            f"{prefix}: zero/negative duration "
            f"(start={segment['start']:.1f}, end={old_end:.1f}), "
            f"set end={segment['end']:.1f}"
        )

    # Fix negative timestamps
    if segment["start"] < 0:
        warnings.append(f"{prefix}: negative start ({segment['start']:.1f}), set to 0.0")
        segment["start"] = 0.0

    # Validate label
    if segment["label"] not in VALID_LABELS:
        warnings.append(
            f"{prefix}: unknown label '{segment['label']}', defaulting to YIELD"
        )
        segment["label"] = "YIELD"

    # Warn on empty text
    if not segment.get("text", "").strip():
        warnings.append(f"{prefix}: empty text")

    return segment, warnings


def convert_file(
    input_path: Path, audio_dir: Path, min_duration: float
) -> Tuple[Optional[Dict], List[str], Dict]:
    """Convert one LLM-processed JSON file to target format.

    Returns:
        (output_data, warnings, original_label_counts)
    """
    warnings = []
    original_counts = Counter()

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audio_file = data.get("audio_file", input_path.stem + ".wav")
    turns = data.get("turns", [])

    if not turns:
        warnings.append(f"  [{input_path.name}]: No turns found, skipping")
        return None, warnings, original_counts

    # Convert turns to segments
    segments = []
    for turn in turns:
        original_label = turn.get("turn_taking_label", "YIELD")
        original_counts[original_label] += 1

        segment = convert_turn_to_segment(turn)
        segment, seg_warnings = validate_and_fix_segment(
            segment, audio_file, min_duration
        )
        warnings.extend(seg_warnings)
        segments.append(segment)

    # Sort by start time
    segments.sort(key=lambda s: (s["start"], s["id"]))

    # Check for overlapping segments (warning only)
    for i in range(1, len(segments)):
        prev_end = segments[i - 1]["end"]
        curr_start = segments[i]["start"]
        if curr_start < prev_end:
            gap = curr_start - prev_end
            warnings.append(
                f"  [{audio_file}] segments {segments[i-1]['id']}->{segments[i]['id']}: "
                f"overlap of {abs(gap):.1f}s"
            )

    # Check audio file existence
    audio_path = audio_dir / audio_file
    audio_found = audio_path.exists()
    if not audio_found:
        warnings.append(f"  [{audio_file}]: audio file not found at {audio_path}")

    # Build output
    output_data = {
        "audio_file": audio_file,
        "segments": segments,
        "metadata": {
            "source_file": input_path.name,
            "processing_method": data.get("processing_method", ""),
            "processing_date": data.get("processing_date", ""),
            "speakers": data.get("speakers", {}),
            "conversion_date": datetime.now().isoformat(),
        },
    }

    # Preserve turn_taking_events for reference
    if "turn_taking_events" in data:
        output_data["metadata"]["turn_taking_events"] = data["turn_taking_events"]

    return output_data, warnings, original_counts


# ==============================================================================
# STATISTICS
# ==============================================================================

def print_statistics(
    results: List[Tuple[str, Optional[Dict], Dict]],
    all_warnings: List[str],
    audio_dir: Path,
):
    """Print comprehensive conversion statistics."""
    total_original = Counter()
    total_merged = Counter()
    file_stats = []

    for filename, output_data, original_counts in results:
        total_original += original_counts

        if output_data is None:
            file_stats.append((filename, 0, Counter()))
            continue

        merged_counts = Counter()
        for seg in output_data["segments"]:
            merged_counts[seg["label"]] += 1
        total_merged += merged_counts
        file_stats.append((filename, len(output_data["segments"]), merged_counts))

    total_segments = sum(total_merged.values())
    total_files = len(results)
    files_with_data = sum(1 for _, n, _ in file_stats if n > 0)

    # Check audio files
    audio_found = 0
    for _, output_data, _ in results:
        if output_data and (audio_dir / output_data["audio_file"]).exists():
            audio_found += 1

    print()
    print("=" * 70)
    print("  05_prepare_dataset.py - Conversion Summary")
    print("=" * 70)
    print()
    print(f"  Input files:  {total_files}")
    print(f"  With data:    {files_with_data}")
    print(f"  Total turns:  {total_segments}")
    print()

    # Before merge
    print("  --- Label Distribution (Before Merge) ---")
    for label in ["YIELD", "HOLD", "BACKCHANNEL", "COOPERATIVE_INTERRUPT", "COMPETITIVE_INTERRUPT"]:
        count = total_original.get(label, 0)
        pct = (count / total_segments * 100) if total_segments > 0 else 0
        print(f"    {label:<30s} {count:>4d}  ({pct:5.1f}%)")
    print(f"    {'Total':<30s} {total_segments:>4d}")
    print()

    # After merge
    print("  --- Label Distribution (After Merge) ---")
    merge_notes = {
        "YIELD": f"  [+{total_original.get('COMPETITIVE_INTERRUPT', 0)} from COMPETITIVE_INTERRUPT]",
        "BACKCHANNEL": f"  [+{total_original.get('COOPERATIVE_INTERRUPT', 0)} from COOPERATIVE_INTERRUPT]",
    }
    for label in ["YIELD", "HOLD", "BACKCHANNEL"]:
        count = total_merged.get(label, 0)
        pct = (count / total_segments * 100) if total_segments > 0 else 0
        note = merge_notes.get(label, "")
        print(f"    {label:<15s} {count:>4d}  ({pct:5.1f}%){note}")
    print(f"    {'Total':<15s} {total_segments:>4d}")
    print()

    # Per-file table
    print("  --- Per-File Summary ---")
    print(f"    {'File':<55s} {'Segs':>5s} {'Y':>4s} {'H':>4s} {'B':>4s}")
    print(f"    {'-'*55} {'-'*5} {'-'*4} {'-'*4} {'-'*4}")
    for filename, n_segs, merged_counts in file_stats:
        short_name = filename[:52] + "..." if len(filename) > 55 else filename
        y = merged_counts.get("YIELD", 0)
        h = merged_counts.get("HOLD", 0)
        b = merged_counts.get("BACKCHANNEL", 0)
        print(f"    {short_name:<55s} {n_segs:>5d} {y:>4d} {h:>4d} {b:>4d}")
    print()

    # Audio check
    print(f"  --- Audio File Check ---")
    print(f"    Found:   {audio_found}/{files_with_data}")
    print(f"    Missing: {files_with_data - audio_found}/{files_with_data}")
    print()

    # Warnings
    if all_warnings:
        print(f"  --- Validation Warnings ({len(all_warnings)}) ---")
        for i, w in enumerate(all_warnings, 1):
            print(f"    [{i}] {w}")
        print()

    print("=" * 70)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert LLM-processed data to training-ready format"
    )
    parser.add_argument(
        "--input", "-i",
        default="datasets/processed/llm_processed",
        help="Input directory with LLM-processed JSON files "
             "(default: datasets/processed/llm_processed)",
    )
    parser.add_argument(
        "--output", "-o",
        default="datasets/processed/final",
        help="Output directory for converted files "
             "(default: datasets/processed/final)",
    )
    parser.add_argument(
        "--audio-dir",
        default="datasets/raw/youtube",
        help="Directory with raw audio files for validation "
             "(default: datasets/raw/youtube)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum segment duration in seconds. "
             "Zero-duration segments are expanded to this. (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing output files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    audio_dir = Path(args.audio_dir)

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # Find all JSON files
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"Error: No JSON files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files in {input_dir}")

    # Convert all files
    results = []
    all_warnings = []

    for json_file in json_files:
        output_data, warnings, original_counts = convert_file(
            json_file, audio_dir, args.min_duration
        )
        results.append((json_file.name, output_data, original_counts))
        all_warnings.extend(warnings)

    # Print statistics
    print_statistics(results, all_warnings, audio_dir)

    # Write output files
    if args.dry_run:
        print("  [DRY RUN] No files written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for filename, output_data, _ in results:
        if output_data is None:
            continue

        output_path = output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        written += 1

    print(f"  Wrote {written} files to {output_dir}")
    print()
    print("  Next step:")
    print(f"    python scripts/06_split_dataset.py "
          f"--input {output_dir} --output datasets/final "
          f"--audio-dir {audio_dir}")


if __name__ == "__main__":
    main()
