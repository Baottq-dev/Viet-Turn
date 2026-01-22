#!/usr/bin/env python3
"""
Script 08: Extract Prosodic Features cho Turn-taking Prediction

Trích xuất các đặc trưng prosody quan trọng cho dự đoán turn-taking:
- F0 (Fundamental Frequency / Pitch)
- Intensity (Cường độ)
- Speaking Rate
- Pause Duration
- Z-score Normalization theo speaker

Các đặc trưng này giúp mô hình phân biệt:
- Giọng lên (rising) vs xuống (falling)
- Thanh điệu từ vựng vs ngữ điệu câu
- Người nói khác nhau (normalization)

Yêu cầu:
    pip install parselmouth numpy torch librosa

Usage:
    python scripts/08_extract_features.py --input data/final --audio-dir data/raw --output data/features
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import parselmouth
    from parselmouth.praat import call
except ImportError as e:
    print("❌ Missing dependencies!")
    print("   Cài đặt: pip install praat-parselmouth numpy")
    print(f"   Error: {e}")
    sys.exit(1)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch not installed. Will save features as .npy instead of .pt")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️ librosa not installed. Using parselmouth for audio loading.")


class ProsodyExtractor:
    """
    Trích xuất đặc trưng prosody từ audio.
    
    Đặc trưng bao gồm:
    - F0 statistics (mean, std, range, slope)
    - Intensity statistics
    - Speaking rate
    - Pause features
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        f0_min: float = 75.0,
        f0_max: float = 500.0,
        frame_length: float = 0.025,  # 25ms
        frame_hop: float = 0.010      # 10ms
    ):
        self.sample_rate = sample_rate
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.frame_length = frame_length
        self.frame_hop = frame_hop
    
    def load_audio(self, audio_path: str) -> parselmouth.Sound:
        """Load audio file as Parselmouth Sound."""
        return parselmouth.Sound(audio_path)
    
    def load_audio_segment(
        self,
        audio_path: str,
        start: float,
        end: float
    ) -> parselmouth.Sound:
        """Load a segment of audio."""
        sound = parselmouth.Sound(audio_path)
        return sound.extract_part(from_time=start, to_time=end)
    
    def extract_f0_features(self, sound: parselmouth.Sound) -> Dict[str, float]:
        """
        Trích xuất F0 (pitch) features.
        
        F0 là tần số cơ bản của giọng nói, quan trọng để:
        - Phân biệt thanh điệu tiếng Việt
        - Nhận diện ngữ điệu (rising/falling intonation)
        """
        pitch = sound.to_pitch(
            time_step=self.frame_hop,
            pitch_floor=self.f0_min,
            pitch_ceiling=self.f0_max
        )
        
        f0_values = pitch.selected_array['frequency']
        f0_voiced = f0_values[f0_values > 0]  # Only voiced frames
        
        if len(f0_voiced) == 0:
            return {
                "f0_mean": 0.0,
                "f0_std": 0.0,
                "f0_min": 0.0,
                "f0_max": 0.0,
                "f0_range": 0.0,
                "f0_slope": 0.0,
                "f0_final": 0.0,
                "voiced_ratio": 0.0
            }
        
        # Basic statistics
        f0_mean = float(np.nanmean(f0_voiced))
        f0_std = float(np.nanstd(f0_voiced))
        f0_min = float(np.nanmin(f0_voiced))
        f0_max = float(np.nanmax(f0_voiced))
        f0_range = f0_max - f0_min
        
        # Slope (rising/falling intonation)
        if len(f0_voiced) > 1:
            x = np.arange(len(f0_voiced))
            slope, _ = np.polyfit(x, f0_voiced, 1)
            f0_slope = float(slope)
        else:
            f0_slope = 0.0
        
        # Final F0 (important for turn-taking)
        # Average of last 3 voiced frames
        f0_final = float(np.mean(f0_voiced[-3:])) if len(f0_voiced) >= 3 else f0_mean
        
        # Voiced ratio
        voiced_ratio = len(f0_voiced) / len(f0_values) if len(f0_values) > 0 else 0.0
        
        return {
            "f0_mean": round(f0_mean, 2),
            "f0_std": round(f0_std, 2),
            "f0_min": round(f0_min, 2),
            "f0_max": round(f0_max, 2),
            "f0_range": round(f0_range, 2),
            "f0_slope": round(f0_slope, 4),
            "f0_final": round(f0_final, 2),
            "voiced_ratio": round(voiced_ratio, 3)
        }
    
    def extract_intensity_features(self, sound: parselmouth.Sound) -> Dict[str, float]:
        """
        Trích xuất Intensity (cường độ) features.
        
        Intensity giúp phân biệt:
        - Backchannel (thường nhỏ) vs. main turn (to hơn)
        - Interruption (tăng đột ngột)
        """
        intensity = sound.to_intensity(
            minimum_pitch=self.f0_min,
            time_step=self.frame_hop
        )
        
        int_values = intensity.values.flatten()
        int_values = int_values[~np.isnan(int_values)]
        
        if len(int_values) == 0:
            return {
                "intensity_mean": 0.0,
                "intensity_std": 0.0,
                "intensity_max": 0.0,
                "intensity_slope": 0.0
            }
        
        int_mean = float(np.mean(int_values))
        int_std = float(np.std(int_values))
        int_max = float(np.max(int_values))
        
        # Intensity slope
        if len(int_values) > 1:
            x = np.arange(len(int_values))
            slope, _ = np.polyfit(x, int_values, 1)
            int_slope = float(slope)
        else:
            int_slope = 0.0
        
        return {
            "intensity_mean": round(int_mean, 2),
            "intensity_std": round(int_std, 2),
            "intensity_max": round(int_max, 2),
            "intensity_slope": round(int_slope, 4)
        }
    
    def extract_temporal_features(self, sound: parselmouth.Sound) -> Dict[str, float]:
        """
        Trích xuất temporal features.
        
        - Speaking rate
        - Duration
        - Pause ratio
        """
        duration = sound.get_total_duration()
        
        # Get intensity to estimate speaking rate
        intensity = sound.to_intensity(minimum_pitch=self.f0_min)
        int_values = intensity.values.flatten()
        
        # Estimate speech frames (above threshold)
        if len(int_values) > 0:
            threshold = np.percentile(int_values[~np.isnan(int_values)], 25)
            speech_frames = np.sum(int_values > threshold)
            total_frames = len(int_values)
            speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
        else:
            speech_ratio = 0
        
        return {
            "duration": round(duration, 3),
            "speech_ratio": round(speech_ratio, 3),
            "speaking_rate_proxy": round(speech_ratio / duration if duration > 0 else 0, 3)
        }
    
    def extract_all_features(self, sound: parselmouth.Sound) -> Dict[str, float]:
        """Extract all prosodic features."""
        features = {}
        features.update(self.extract_f0_features(sound))
        features.update(self.extract_intensity_features(sound))
        features.update(self.extract_temporal_features(sound))
        return features
    
    def extract_for_segment(
        self,
        audio_path: str,
        start: float,
        end: float
    ) -> Dict[str, float]:
        """Extract features for a specific segment."""
        sound = self.load_audio_segment(audio_path, start, end)
        return self.extract_all_features(sound)


class SpeakerNormalizer:
    """
    Z-score normalization theo speaker.
    
    Giúp mô hình học được pattern chung thay vì đặc điểm riêng
    của từng người nói (giọng cao/thấp, to/nhỏ).
    """
    
    def __init__(self):
        self.speaker_stats = {}
    
    def compute_speaker_stats(
        self,
        features_by_segment: List[Dict],
        speaker_key: str = "speaker"
    ):
        """Compute mean và std cho mỗi speaker."""
        by_speaker = {}
        
        for item in features_by_segment:
            speaker = item.get(speaker_key, "UNKNOWN")
            if speaker not in by_speaker:
                by_speaker[speaker] = []
            by_speaker[speaker].append(item.get("features", {}))
        
        # Compute stats for each speaker
        for speaker, features_list in by_speaker.items():
            if not features_list:
                continue
            
            # Get all numeric features
            feature_keys = [k for k in features_list[0].keys() 
                          if isinstance(features_list[0][k], (int, float))]
            
            stats = {}
            for key in feature_keys:
                values = [f[key] for f in features_list if key in f]
                if values:
                    stats[key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)) if len(values) > 1 else 1.0
                    }
            
            self.speaker_stats[speaker] = stats
    
    def normalize(
        self,
        features: Dict[str, float],
        speaker: str
    ) -> Dict[str, float]:
        """Apply Z-score normalization."""
        if speaker not in self.speaker_stats:
            return features
        
        stats = self.speaker_stats[speaker]
        normalized = {}
        
        for key, value in features.items():
            if key in stats and stats[key]["std"] > 0:
                z_score = (value - stats[key]["mean"]) / stats[key]["std"]
                normalized[f"{key}_zscore"] = round(z_score, 4)
            normalized[key] = value
        
        return normalized


def process_split(
    split_file: str,
    audio_dir: str,
    output_dir: str,
    extractor: ProsodyExtractor,
    save_format: str = "pt"
) -> Tuple[int, int]:
    """
    Process một split file và extract features.
    
    Returns:
        Tuple of (success_count, failed_count)
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
    
    success = 0
    failed = 0
    all_features = []
    
    for audio_file, segs in by_audio.items():
        # Find audio
        audio_path = None
        for search_dir in [audio_dir, Path(audio_dir) / "youtube"]:
            candidate = Path(search_dir) / audio_file
            if candidate.exists():
                audio_path = str(candidate)
                break
        
        if not audio_path:
            failed += len(segs)
            continue
        
        # Extract features for each segment
        for seg in segs:
            try:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                
                if end - start < 0.1:  # Too short
                    failed += 1
                    continue
                
                features = extractor.extract_for_segment(audio_path, start, end)
                
                # Add segment info
                features["segment_id"] = seg.get("id", 0)
                features["speaker"] = seg.get("speaker", "UNKNOWN")
                features["label"] = seg.get("label", seg.get("auto_label", "UNKNOWN"))
                
                all_features.append({
                    "segment": seg,
                    "features": features,
                    "speaker": features["speaker"]
                })
                
                success += 1
                
            except Exception as e:
                print(f"   ⚠️ Error extracting features: {e}")
                failed += 1
    
    # Apply speaker normalization
    if all_features:
        print(f"   🔄 Applying Z-score normalization...")
        normalizer = SpeakerNormalizer()
        normalizer.compute_speaker_stats(all_features)
        
        for item in all_features:
            item["features"] = normalizer.normalize(
                item["features"],
                item["speaker"]
            )
    
    # Save features
    split_name = Path(split_file).stem
    
    if save_format == "pt" and HAS_TORCH:
        # Save as PyTorch tensors
        features_data = {
            "segments": [item["segment"] for item in all_features],
            "features": [item["features"] for item in all_features],
            "feature_names": list(all_features[0]["features"].keys()) if all_features else []
        }
        
        # Convert to tensors
        feature_matrix = []
        for item in all_features:
            row = [item["features"].get(k, 0) for k in features_data["feature_names"]
                   if isinstance(item["features"].get(k, 0), (int, float))]
            feature_matrix.append(row)
        
        if feature_matrix:
            features_data["feature_tensor"] = torch.tensor(feature_matrix, dtype=torch.float32)
        
        output_file = output_path / f"{split_name}_features.pt"
        torch.save(features_data, output_file)
    else:
        # Save as JSON
        features_data = {
            "segments": [item["segment"] for item in all_features],
            "features": [item["features"] for item in all_features]
        }
        
        output_file = output_path / f"{split_name}_features.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(features_data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Saved: {output_file}")
    
    return success, failed


def main():
    parser = argparse.ArgumentParser(
        description="Extract Prosodic Features cho Turn-taking Prediction"
    )
    
    parser.add_argument("--input", "-i", required=True,
                        help="Input dir chứa train.json, val.json, test.json")
    parser.add_argument("--audio-dir", "-a", required=True,
                        help="Thư mục chứa audio gốc")
    parser.add_argument("--output", "-o", default="data/features",
                        help="Output dir cho features")
    parser.add_argument("--format", choices=["pt", "json"], default="pt",
                        help="Output format (default: pt)")
    parser.add_argument("--f0-min", type=float, default=75.0,
                        help="Minimum F0 (Hz)")
    parser.add_argument("--f0-max", type=float, default=500.0,
                        help="Maximum F0 (Hz)")
    
    args = parser.parse_args()
    
    # Init extractor
    extractor = ProsodyExtractor(
        f0_min=args.f0_min,
        f0_max=args.f0_max
    )
    
    input_path = Path(args.input)
    
    total_success = 0
    total_failed = 0
    
    print(f"🎵 Extracting prosodic features...")
    print(f"   F0 range: {args.f0_min}-{args.f0_max} Hz")
    print(f"   Output format: {args.format}")
    
    for split_name in ["train", "val", "test"]:
        split_file = input_path / f"{split_name}.json"
        
        if not split_file.exists():
            print(f"⏭️  {split_name}.json not found, skipping")
            continue
        
        print(f"\n📂 Processing {split_name}...")
        
        success, failed = process_split(
            str(split_file),
            args.audio_dir,
            args.output,
            extractor,
            args.format
        )
        
        total_success += success
        total_failed += failed
        
        print(f"   Success: {success}, Failed: {failed}")
    
    print(f"\n📊 Summary:")
    print(f"   Total extracted: {total_success}")
    print(f"   Total failed: {total_failed}")
    print(f"   Output: {args.output}")
    
    print(f"\n💡 Features included:")
    print(f"   - F0: mean, std, range, slope, final")
    print(f"   - Intensity: mean, std, max, slope")
    print(f"   - Temporal: duration, speech_ratio")
    print(f"   - Z-score normalized by speaker")


if __name__ == "__main__":
    main()
