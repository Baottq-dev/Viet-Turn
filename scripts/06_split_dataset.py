#!/usr/bin/env python3
"""
Script 06: Split dataset thành Train/Val/Test

Chia dataset và extract audio features cho training.

Usage:
    python scripts/06_split_dataset.py --input data/processed/final --output data/final
    python scripts/06_split_dataset.py --input data/processed/final --output data/final --extract-features
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import sys

# Encoding fix for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_all_segments(input_dir: str) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    """
    Load tất cả segments từ các JSON files.
    
    Returns:
        (file_to_segments dict, all_segments list)
    """
    input_path = Path(input_dir)
    
    file_segments = {}
    all_segments = []
    
    for json_file in sorted(input_path.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        segments = data.get("segments", [])
        
        # Add source info to each segment
        for seg in segments:
            seg["source_file"] = json_file.name
            seg["audio_file"] = data.get("audio_file", json_file.stem + ".wav")
        
        file_segments[json_file.name] = segments
        all_segments.extend(segments)
    
    return file_segments, all_segments


def split_by_file(
    file_segments: Dict[str, List[Dict]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split theo file (không split trong file để tránh data leakage).
    
    Returns:
        (train_segments, val_segments, test_segments)
    """
    random.seed(seed)
    
    files = list(file_segments.keys())
    random.shuffle(files)
    
    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    
    train_segments = []
    val_segments = []
    test_segments = []
    
    for f in train_files:
        train_segments.extend(file_segments[f])
    for f in val_files:
        val_segments.extend(file_segments[f])
    for f in test_files:
        test_segments.extend(file_segments[f])
    
    return train_segments, val_segments, test_segments


def compute_stats(segments: List[Dict]) -> Dict:
    """Compute statistics cho một split"""
    if not segments:
        return {}
    
    labels = [s.get("label", "UNKNOWN") for s in segments]
    label_counts = Counter(labels)
    
    reviewed = sum(1 for s in segments if s.get("reviewed", False))
    
    return {
        "num_segments": len(segments),
        "num_files": len(set(s.get("source_file") for s in segments)),
        "label_distribution": dict(label_counts),
        "reviewed_ratio": reviewed / len(segments) if segments else 0
    }


def save_split(
    segments: List[Dict],
    output_path: Path,
    name: str
) -> str:
    """Save một split"""
    output_file = output_path / f"{name}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    
    return str(output_file)


def extract_features_for_split(
    segments: List[Dict],
    audio_dir: str,
    output_dir: str,
    split_name: str
):
    """Extract audio features cho một split"""
    try:
        from src.data.audio_processor import AudioProcessor
        import torch
        import librosa
    except ImportError as e:
        print(f"   ⚠️  Cannot import: {e}")
        print("       Skipping feature extraction")
        return
    
    processor = AudioProcessor()
    features_dir = Path(output_dir) / "features" / split_name
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by audio file
    file_segments = {}
    for seg in segments:
        audio_file = seg.get("audio_file", "")
        if audio_file not in file_segments:
            file_segments[audio_file] = []
        file_segments[audio_file].append(seg)
    
    print(f"   Extracting features for {len(file_segments)} audio files...")
    
    for audio_name, segs in file_segments.items():
        audio_path = Path(audio_dir) / audio_name
        
        if not audio_path.exists():
            print(f"      ⚠️  Audio not found: {audio_path}")
            continue
        
        try:
            # Load full audio
            audio, sr = librosa.load(str(audio_path), sr=16000)
            
            for seg in segs:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                
                # Extract segment
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                segment_audio = audio[start_sample:end_sample]
                
                if len(segment_audio) < 1600:  # < 0.1s
                    continue
                
                # Extract features
                features = processor(segment_audio)
                
                # Save
                feature_file = features_dir / f"{Path(audio_name).stem}_{seg['id']}.pt"
                torch.save(features, feature_file)
                
                seg["feature_file"] = str(feature_file)
        
        except Exception as e:
            print(f"      ⚠️  Error processing {audio_name}: {e}")


def create_manifest(
    train_segments: List[Dict],
    val_segments: List[Dict],
    test_segments: List[Dict],
    output_dir: str
):
    """Create manifest file với metadata"""
    manifest = {
        "version": "1.0",
        "splits": {
            "train": {
                "file": "train.json",
                "stats": compute_stats(train_segments)
            },
            "val": {
                "file": "val.json", 
                "stats": compute_stats(val_segments)
            },
            "test": {
                "file": "test.json",
                "stats": compute_stats(test_segments)
            }
        },
        "total_segments": len(train_segments) + len(val_segments) + len(test_segments),
        "labels": ["YIELD", "HOLD", "BACKCHANNEL"]
    }
    
    manifest_file = Path(output_dir) / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset và prepare cho training"
    )
    
    parser.add_argument("--input", "-i", required=True, help="Input dir (merged data)")
    parser.add_argument("--output", "-o", default="data/final", help="Output dir")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--extract-features", action="store_true", help="Extract audio features")
    parser.add_argument("--audio-dir", default="data/raw/youtube", help="Audio source dir")
    
    args = parser.parse_args()
    
    test_ratio = 1 - args.train_ratio - args.val_ratio
    
    print(f"📂 Loading data from {args.input}")
    file_segments, all_segments = load_all_segments(args.input)
    
    print(f"   Found {len(all_segments)} segments from {len(file_segments)} files")
    
    # Split
    print(f"\n🔀 Splitting: {args.train_ratio:.0%} train, {args.val_ratio:.0%} val, {test_ratio:.0%} test")
    train, val, test = split_by_file(
        file_segments,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving splits to {args.output}")
    save_split(train, output_path, "train")
    save_split(val, output_path, "val")
    save_split(test, output_path, "test")
    
    # Extract features if requested
    if args.extract_features:
        print(f"\n🎵 Extracting audio features...")
        extract_features_for_split(train, args.audio_dir, args.output, "train")
        extract_features_for_split(val, args.audio_dir, args.output, "val")
        extract_features_for_split(test, args.audio_dir, args.output, "test")
    
    # Create manifest
    manifest = create_manifest(train, val, test, args.output)
    
    # Print summary
    print(f"\n📊 Dataset Summary:")
    print(f"   Train: {manifest['splits']['train']['stats']}")
    print(f"   Val:   {manifest['splits']['val']['stats']}")
    print(f"   Test:  {manifest['splits']['test']['stats']}")
    
    print(f"\n✅ Dataset ready at {args.output}")
    print(f"   - train.json: {len(train)} segments")
    print(f"   - val.json: {len(val)} segments")
    print(f"   - test.json: {len(test)} segments")
    print(f"   - manifest.json: metadata")


if __name__ == "__main__":
    main()
