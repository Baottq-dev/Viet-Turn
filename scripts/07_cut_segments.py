#!/usr/bin/env python3
"""
Script 07: Cắt audio thành segments riêng biệt

Cắt file audio 1 tiếng thành các file nhỏ 2-10 giây theo timestamps.

Usage:
    python scripts/07_cut_segments.py --input data/final --audio-dir data/raw --output data/segments
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

try:
    import librosa
    import soundfile as sf
    import numpy as np
except ImportError as e:
    print("❌ Missing dependencies!")
    print("   Cài đặt: pip install librosa soundfile")
    print(f"   Error: {e}")
    sys.exit(1)


def cut_audio_segment(
    audio_path: str,
    start: float,
    end: float,
    output_path: str,
    sample_rate: int = 16000,
    min_duration: float = 0.3
) -> bool:
    """
    Cắt một đoạn audio từ file gốc.
    
    Args:
        audio_path: Path to source audio
        start: Start time in seconds
        end: End time in seconds  
        output_path: Output file path
        sample_rate: Target sample rate
        min_duration: Minimum segment duration
        
    Returns:
        True if successful
    """
    duration = end - start
    if duration < min_duration:
        return False
    
    try:
        # Load only the segment we need (efficient for large files)
        audio, sr = librosa.load(
            audio_path,
            sr=sample_rate,
            offset=start,
            duration=duration
        )
        
        # Save
        sf.write(output_path, audio, sample_rate)
        return True
        
    except Exception as e:
        print(f"   ⚠️ Error cutting segment: {e}")
        return False


def cut_segments_from_json(
    json_path: str,
    audio_dir: str,
    output_dir: str,
    sample_rate: int = 16000,
    min_duration: float = 0.3
) -> tuple:
    """
    Cắt tất cả segments từ một JSON file.
    
    Returns:
        Tuple of (data dict, cut_count, skip_count)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    audio_file = data.get("audio_file", "")
    audio_path = Path(audio_dir) / audio_file
    
    if not audio_path.exists():
        print(f"   ⚠️ Audio not found: {audio_path}")
        return data
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(json_path).stem
    
    cut_count = 0
    skip_count = 0
    
    for segment in data.get("segments", []):
        seg_id = segment.get("id", 0)
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        
        # Output filename
        seg_filename = f"{base_name}_{seg_id:04d}.wav"
        seg_output = output_path / seg_filename
        
        if seg_output.exists():
            # Already cut
            segment["audio_segment"] = str(seg_output)
            cut_count += 1
            continue
        
        # Cut segment
        if cut_audio_segment(str(audio_path), start, end, str(seg_output), sample_rate, min_duration):
            segment["audio_segment"] = str(seg_output)
            cut_count += 1
        else:
            skip_count += 1
    
    return data, cut_count, skip_count


def process_split(
    split_file: str,
    audio_dir: str,
    output_dir: str,
    sample_rate: int = 16000
) -> Dict:
    """
    Process một split file (train.json, val.json, test.json)
    """
    with open(split_file, "r", encoding="utf-8") as f:
        segments = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group by audio file
    by_audio = {}
    for seg in segments:
        audio_file = seg.get("audio_file", "")
        if audio_file not in by_audio:
            by_audio[audio_file] = []
        by_audio[audio_file].append(seg)
    
    print(f"   Processing {len(by_audio)} audio files...")
    
    total_cut = 0
    total_skip = 0
    
    for audio_file, segs in by_audio.items():
        audio_path = Path(audio_dir) / audio_file
        
        if not audio_path.exists():
            # Try subdirectories
            for subdir in Path(audio_dir).iterdir():
                if subdir.is_dir():
                    candidate = subdir / audio_file
                    if candidate.exists():
                        audio_path = candidate
                        break
        
        if not audio_path.exists():
            print(f"   ⚠️ Audio not found: {audio_file}")
            total_skip += len(segs)
            continue
        
        # Load audio once
        try:
            audio, sr = librosa.load(str(audio_path), sr=sample_rate)
        except Exception as e:
            print(f"   ⚠️ Error loading {audio_file}: {e}")
            continue
        
        base_name = Path(audio_file).stem
        
        for seg in segs:
            seg_id = seg.get("id", 0)
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            
            # Calculate samples
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            
            # Skip too short
            if end_sample - start_sample < int(0.3 * sample_rate):
                total_skip += 1
                continue
            
            # Extract segment
            segment_audio = audio[start_sample:end_sample]
            
            # Save
            seg_filename = f"{base_name}_{seg_id:04d}.wav"
            seg_output = output_path / seg_filename
            
            if not seg_output.exists():
                sf.write(str(seg_output), segment_audio, sample_rate)
            
            seg["audio_segment"] = str(seg_output)
            total_cut += 1
    
    # Save updated split
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    
    return total_cut, total_skip


def main():
    parser = argparse.ArgumentParser(
        description="Cắt audio thành segments riêng biệt"
    )
    
    parser.add_argument("--input", "-i", required=True, 
                        help="Input dir chứa train.json, val.json, test.json")
    parser.add_argument("--audio-dir", "-a", required=True,
                        help="Thư mục chứa audio gốc")
    parser.add_argument("--output", "-o", default="data/segments",
                        help="Output dir cho segments")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Sample rate (default: 16000)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    total_cut = 0
    total_skip = 0
    
    for split_name in ["train", "val", "test"]:
        split_file = input_path / f"{split_name}.json"
        
        if not split_file.exists():
            print(f"⏭️  {split_name}.json not found, skipping")
            continue
        
        print(f"\n📂 Processing {split_name}...")
        
        output_dir = Path(args.output) / split_name
        cut, skip = process_split(
            str(split_file),
            args.audio_dir,
            str(output_dir),
            args.sample_rate
        )
        
        total_cut += cut
        total_skip += skip
        
        print(f"   ✅ Cut: {cut}, Skipped: {skip}")
    
    print(f"\n📊 Summary:")
    print(f"   Total segments cut: {total_cut}")
    print(f"   Total skipped: {total_skip}")
    print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
