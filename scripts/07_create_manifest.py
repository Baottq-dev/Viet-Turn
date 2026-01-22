#!/usr/bin/env python3
"""
Script 07: Tạo Manifest cho VAP Training (Thay thế cắt đoạn)

Thay vì cắt file audio thành các segment riêng lẻ, script này tạo manifest
file với timestamps để hỗ trợ sliding window training cho Voice Activity
Projection (VAP) models.

Lợi ích:
- Giữ nguyên file audio gốc
- Bảo toàn context lịch sử (inter-turn silence)
- Hỗ trợ sliding window cho VAP/TurnGPT training

Usage:
    python scripts/07_create_manifest.py --input data/final --audio-dir data/raw --output data/manifest
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    import librosa
except ImportError:
    print("❌ Missing dependency: pip install librosa")
    sys.exit(1)


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        duration = librosa.get_duration(path=audio_path)
        return round(duration, 2)
    except Exception as e:
        print(f"   ⚠️ Error getting duration for {audio_path}: {e}")
        return 0.0


def create_vap_events(segments: List[Dict]) -> List[Dict]:
    """
    Convert segments to VAP-style events.
    
    Each event represents a turn-taking point with context window info.
    """
    events = []
    
    for i, seg in enumerate(segments):
        # Calculate context window
        context_start = max(0, seg.get("start", 0) - 10.0)  # 10s history
        
        # Get previous segment end for inter-turn silence calculation
        prev_end = segments[i-1].get("end", 0) if i > 0 else 0
        inter_turn_silence = seg.get("start", 0) - prev_end
        
        event = {
            "id": seg.get("id", i),
            "time": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "duration": round(seg.get("end", 0) - seg.get("start", 0), 2),
            "type": seg.get("label", seg.get("auto_label", "YIELD")),
            "speaker": seg.get("speaker", "UNKNOWN"),
            "text": seg.get("text", ""),
            
            # VAP-specific fields
            "context_start": round(context_start, 2),
            "inter_turn_silence": round(inter_turn_silence, 2),
            
            # Overlap info (if available)
            "has_overlap": seg.get("has_overlap", False),
            "overlap_duration": seg.get("overlap_duration", 0),
            
            # Review status
            "reviewed": seg.get("reviewed", False),
            "confidence": seg.get("confidence", 0),
        }
        
        events.append(event)
    
    return events


def create_manifest_for_file(
    json_path: str,
    audio_dir: str,
    vap_config: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Create manifest for a single labeled JSON file.
    
    Args:
        json_path: Path to labeled JSON file
        audio_dir: Directory containing audio files
        vap_config: VAP training configuration
        
    Returns:
        Manifest dict or None if audio not found
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    audio_file = data.get("audio_file", Path(json_path).stem + ".wav")
    
    # Find audio file
    audio_path = None
    for search_dir in [audio_dir, Path(audio_dir) / "youtube", Path(audio_dir).parent]:
        candidate = Path(search_dir) / audio_file
        if candidate.exists():
            audio_path = candidate
            break
    
    if not audio_path or not audio_path.exists():
        return None
    
    # Get actual duration
    duration = get_audio_duration(str(audio_path))
    
    # Convert segments to VAP events
    segments = data.get("segments", [])
    events = create_vap_events(segments)
    
    # Default VAP config
    if vap_config is None:
        vap_config = {
            "history_window": 10.0,      # 10 seconds of history
            "prediction_window": 2.0,     # Predict 2 seconds ahead
            "hop_length": 0.1,            # 100ms sliding window hop
            "sample_rate": 16000,
        }
    
    # Create manifest
    manifest = {
        "file_id": Path(json_path).stem,
        "audio_path": str(audio_path.absolute()),
        "audio_file": audio_file,
        "duration": duration,
        "num_events": len(events),
        "events": events,
        "vap_config": vap_config,
        
        # Stats
        "label_distribution": get_label_distribution(events),
        "speakers": list(set(e["speaker"] for e in events)),
        "reviewed_ratio": sum(1 for e in events if e["reviewed"]) / len(events) if events else 0,
        
        # Metadata
        "source_json": str(Path(json_path).absolute()),
        "created_at": datetime.now().isoformat(),
    }
    
    return manifest


def get_label_distribution(events: List[Dict]) -> Dict[str, int]:
    """Count events by label type."""
    dist = {}
    for e in events:
        label = e.get("type", "UNKNOWN")
        dist[label] = dist.get(label, 0) + 1
    return dist


def create_split_manifest(
    split_file: str,
    audio_dir: str,
    vap_config: Optional[Dict] = None
) -> Tuple[List[Dict], Dict]:
    """
    Create manifest for a split file (train.json, val.json, test.json).
    
    Returns:
        Tuple of (manifests list, stats dict)
    """
    with open(split_file, "r", encoding="utf-8") as f:
        segments = json.load(f)
    
    # Group segments by audio file
    by_audio = {}
    for seg in segments:
        audio_file = seg.get("audio_file", "")
        if audio_file not in by_audio:
            by_audio[audio_file] = []
        by_audio[audio_file].append(seg)
    
    manifests = []
    stats = {"found": 0, "missing": 0, "total_events": 0}
    
    for audio_file, segs in by_audio.items():
        # Find audio
        audio_path = None
        for search_dir in [audio_dir, Path(audio_dir) / "youtube"]:
            candidate = Path(search_dir) / audio_file
            if candidate.exists():
                audio_path = candidate
                break
        
        if not audio_path:
            stats["missing"] += 1
            continue
        
        # Create manifest for this audio
        duration = get_audio_duration(str(audio_path))
        events = create_vap_events(segs)
        
        manifest = {
            "file_id": Path(audio_file).stem,
            "audio_path": str(audio_path.absolute()),
            "audio_file": audio_file,
            "duration": duration,
            "num_events": len(events),
            "events": events,
            "vap_config": vap_config or {
                "history_window": 10.0,
                "prediction_window": 2.0,
                "hop_length": 0.1,
                "sample_rate": 16000,
            },
            "label_distribution": get_label_distribution(events),
        }
        
        manifests.append(manifest)
        stats["found"] += 1
        stats["total_events"] += len(events)
    
    return manifests, stats


def create_dataloader_manifest(
    manifests: List[Dict],
    split_name: str,
    output_dir: str
) -> str:
    """
    Create a combined manifest for PyTorch DataLoader.
    
    This manifest contains all audio files and their events,
    suitable for random sliding window sampling during training.
    """
    # Aggregate all events with file references
    all_samples = []
    
    for manifest in manifests:
        audio_path = manifest["audio_path"]
        duration = manifest["duration"]
        vap_config = manifest["vap_config"]
        
        for event in manifest["events"]:
            sample = {
                "audio_path": audio_path,
                "audio_duration": duration,
                "event": event,
                "vap_config": vap_config,
            }
            all_samples.append(sample)
    
    # Create dataloader manifest
    dl_manifest = {
        "split": split_name,
        "num_files": len(manifests),
        "num_samples": len(all_samples),
        "samples": all_samples,
        "vap_config": manifests[0]["vap_config"] if manifests else {},
        "created_at": datetime.now().isoformat(),
    }
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    manifest_file = output_path / f"{split_name}_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(dl_manifest, f, ensure_ascii=False, indent=2)
    
    return str(manifest_file)


def main():
    parser = argparse.ArgumentParser(
        description="Tạo Manifest cho VAP Training (Thay thế cắt đoạn)"
    )
    
    parser.add_argument("--input", "-i", required=True,
                        help="Input dir chứa train.json, val.json, test.json")
    parser.add_argument("--audio-dir", "-a", required=True,
                        help="Thư mục chứa audio gốc")
    parser.add_argument("--output", "-o", default="data/manifest",
                        help="Output dir cho manifest files")
    parser.add_argument("--history-window", type=float, default=10.0,
                        help="History window size in seconds (default: 10.0)")
    parser.add_argument("--prediction-window", type=float, default=2.0,
                        help="Prediction window size in seconds (default: 2.0)")
    
    args = parser.parse_args()
    
    vap_config = {
        "history_window": args.history_window,
        "prediction_window": args.prediction_window,
        "hop_length": 0.1,
        "sample_rate": 16000,
    }
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_stats = {"files": 0, "events": 0, "missing": 0}
    
    for split_name in ["train", "val", "test"]:
        split_file = input_path / f"{split_name}.json"
        
        if not split_file.exists():
            print(f"⏭️  {split_name}.json not found, skipping")
            continue
        
        print(f"\n📂 Processing {split_name}...")
        
        manifests, stats = create_split_manifest(
            str(split_file),
            args.audio_dir,
            vap_config
        )
        
        if manifests:
            # Save individual manifests
            split_output = output_path / split_name
            split_output.mkdir(exist_ok=True)
            
            for manifest in manifests:
                manifest_file = split_output / f"{manifest['file_id']}.json"
                with open(manifest_file, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)
            
            # Create dataloader manifest
            dl_manifest = create_dataloader_manifest(manifests, split_name, str(output_path))
            
            print(f"   ✅ Files: {stats['found']}, Events: {stats['total_events']}")
            print(f"   📄 Manifest: {dl_manifest}")
            
            if stats["missing"] > 0:
                print(f"   ⚠️  Missing audio: {stats['missing']}")
        
        total_stats["files"] += stats["found"]
        total_stats["events"] += stats["total_events"]
        total_stats["missing"] += stats["missing"]
    
    print(f"\n📊 Summary:")
    print(f"   Total files: {total_stats['files']}")
    print(f"   Total events: {total_stats['events']}")
    print(f"   Missing audio: {total_stats['missing']}")
    print(f"   Output: {args.output}")
    
    print(f"\n💡 VAP Training Config:")
    print(f"   History window: {vap_config['history_window']}s")
    print(f"   Prediction window: {vap_config['prediction_window']}s")
    print(f"   Sample rate: {vap_config['sample_rate']}Hz")
    
    print(f"\n🚀 Next: Use manifest with VAP DataLoader for sliding window training")


if __name__ == "__main__":
    main()
