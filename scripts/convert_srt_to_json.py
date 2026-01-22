#!/usr/bin/env python3
"""
Script: Convert SRT subtitles to pipeline JSON format

Usage:
    python scripts/convert_srt_to_json.py --input datasets/dataset-youtube-sub/sub --output datasets/processed/srt
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def parse_srt_time(time_str: str) -> float:
    """Convert SRT timestamp to seconds."""
    # Format: HH:MM:SS,mmm or HH:MM:SS.mmm
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_srt(srt_content: str) -> List[Dict]:
    """Parse SRT content into list of segments."""
    segments = []
    
    # Split by double newline (segment separator)
    blocks = re.split(r'\n\n+', srt_content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        
        # First line is segment number
        try:
            seg_id = int(lines[0].strip())
        except ValueError:
            continue
        
        # Second line is timestamp
        time_match = re.match(
            r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})',
            lines[1].strip()
        )
        if not time_match:
            continue
        
        start_time = parse_srt_time(time_match.group(1))
        end_time = parse_srt_time(time_match.group(2))
        
        # Remaining lines are text
        text = ' '.join(lines[2:]).strip()
        
        if text:  # Only add if there's text
            segments.append({
                "id": seg_id - 1,  # 0-indexed
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "text": text,
                "speaker": "UNKNOWN",  # SRT has no speaker info
                "words": []
            })
    
    return segments


def merge_into_sentences(
    segments: List[Dict], 
    silence_gap_threshold: float = 0.8,
    max_segment_duration: float = 30.0
) -> List[Dict]:
    """
    Merge short segments into sentence-like units based on:
    1. Sentence-ending punctuation (. ? ! ...)
    2. Silence gaps (> threshold = new segment)
    3. Maximum segment duration
    
    Args:
        segments: List of parsed SRT segments
        silence_gap_threshold: Gap in seconds that indicates a new turn/sentence
        max_segment_duration: Maximum duration for merged segment
    
    Returns:
        List of merged segments
    """
    if not segments:
        return []
    
    # Sentence-ending patterns
    sentence_end_pattern = re.compile(r'[.!?…।]$|[.!?…।]\s*$')
    
    merged = []
    current = segments[0].copy()
    
    for seg in segments[1:]:
        gap = seg["start"] - current["end"]
        current_duration = current["end"] - current["start"]
        merged_duration = seg["end"] - current["start"]
        
        # Check if we should start a new segment
        should_split = False
        
        # 1. Large silence gap = new turn/segment
        if gap > silence_gap_threshold:
            should_split = True
        
        # 2. Previous segment ends with sentence punctuation
        elif sentence_end_pattern.search(current["text"].strip()):
            should_split = True
        
        # 3. Merged would be too long
        elif merged_duration > max_segment_duration:
            should_split = True
        
        if should_split:
            merged.append(current)
            current = seg.copy()
        else:
            # Merge: extend current segment
            current["end"] = seg["end"]
            current["text"] = current["text"].rstrip() + " " + seg["text"].lstrip()
    
    # Don't forget the last segment
    merged.append(current)
    
    # Re-index
    for i, seg in enumerate(merged):
        seg["id"] = i
    
    return merged


def merge_short_segments(segments: List[Dict], min_duration: float = 2.0) -> List[Dict]:
    """Legacy merge function - use merge_into_sentences instead."""
    return merge_into_sentences(segments)


def convert_srt_to_json(srt_path: str, output_dir: str, merge_segments: bool = True) -> Dict:
    """Convert SRT file to pipeline JSON format."""
    srt_file = Path(srt_path)
    
    # Read SRT file
    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    
    # Parse SRT
    segments = parse_srt(srt_content)
    
    # Optionally merge short segments
    if merge_segments:
        segments = merge_short_segments(segments)
    
    # Calculate duration
    duration = segments[-1]["end"] if segments else 0
    
    # Create output
    output_data = {
        "source_file": srt_file.name,
        "source_type": "youtube_subtitle",
        "duration": round(duration, 2),
        "num_segments": len(segments),
        "segments": segments
    }
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean filename
    output_name = re.sub(r'\[.*?\]', '', srt_file.stem).strip()
    output_name = re.sub(r'[^\w\s-]', '', output_name).strip()
    output_name = re.sub(r'\s+', '_', output_name)[:100]  # Limit length
    
    json_path = output_path / f"{output_name}.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Converted: {srt_file.name}")
    print(f"   → {json_path}")
    print(f"   Segments: {len(segments)}, Duration: {duration:.0f}s")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Convert SRT subtitles to pipeline JSON")
    parser.add_argument("--input", required=True, help="Input SRT file or directory")
    parser.add_argument("--output", default="datasets/processed/srt", help="Output directory")
    parser.add_argument("--no-merge", action="store_true", help="Don't merge short segments")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        convert_srt_to_json(str(input_path), args.output, merge_segments=not args.no_merge)
    elif input_path.is_dir():
        # Directory
        srt_files = list(input_path.glob("*.srt"))
        print(f"📂 Found {len(srt_files)} SRT files")
        
        for srt_file in srt_files:
            try:
                convert_srt_to_json(str(srt_file), args.output, merge_segments=not args.no_merge)
            except Exception as e:
                print(f"❌ Error: {srt_file.name}: {e}")
        
        print(f"\n📊 Done! Converted {len(srt_files)} files")
    else:
        print(f"❌ Not found: {args.input}")


if __name__ == "__main__":
    main()
