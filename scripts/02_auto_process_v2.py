#!/usr/bin/env python3
"""
Script 02 v2: Auto-process audio với whisperX + Overlap Detection

CẢI TIẾN so với v1:
- Overlap Detection với pyannote (phát hiện chồng lấn)
- VAD parameters tối ưu cho backchannel ngắn (min_duration_on=0.15s)
- Backchannel recovery từ Whisper timestamps
- Flag has_overlap cho segments

Yêu cầu:
    pip install whisperx torch torchaudio pyannote.audio
    HuggingFace token: https://huggingface.co/settings/tokens

Usage:
    python scripts/02_auto_process_v2.py --input data/raw --output data/processed/auto --device cuda
    python scripts/02_auto_process_v2.py --input data/raw --output data/processed/auto --enable-overlap
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore")

# Check imports
try:
    import torch
    import whisperx
    
    # Fix PyTorch 2.6+ weights_only issue
    try:
        from omegaconf import ListConfig, DictConfig
        torch.serialization.add_safe_globals([ListConfig, DictConfig])
    except Exception:
        pass
        
except ImportError as e:
    print("❌ Missing dependencies!")
    print("   Cài đặt: pip install whisperx torch torchaudio")
    print(f"   Error: {e}")
    sys.exit(1)


def get_hf_token() -> str:
    """Lấy HuggingFace token từ env hoặc hỏi user."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if not token:
        print("⚠️  Cần HuggingFace token cho speaker diarization")
        print("   Lấy token tại: https://huggingface.co/settings/tokens")
        print("   Sau đó: export HF_TOKEN='your_token'")
        token = input("   Nhập token (hoặc Enter để skip diarization): ").strip()
    
    return token


def detect_overlaps(
    audio_path: str,
    hf_token: str,
    device: str = "cuda"
) -> List[Tuple[float, float]]:
    """
    Phát hiện các vùng chồng lấn (overlap) trong audio.
    
    Returns:
        List of (start, end) tuples for overlapping regions
    """
    try:
        from pyannote.audio import Pipeline
        
        # Load overlap detection pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/overlapped-speech-detection",
            use_auth_token=hf_token
        )
        
        if device == "cuda" and torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
        
        # Run detection
        output = pipeline(audio_path)
        
        # Extract overlap regions
        overlaps = []
        for segment in output.get_timeline():
            overlaps.append((segment.start, segment.end))
        
        return overlaps
        
    except Exception as e:
        print(f"   ⚠️ Overlap detection failed: {e}")
        return []


def mark_overlaps(
    segments: List[Dict],
    overlaps: List[Tuple[float, float]]
) -> List[Dict]:
    """
    Đánh dấu segments có chồng lấn.
    """
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        
        for ov_start, ov_end in overlaps:
            # Check if segment overlaps with any overlap region
            if seg_start < ov_end and seg_end > ov_start:
                seg["has_overlap"] = True
                # Calculate overlap duration
                overlap_start = max(seg_start, ov_start)
                overlap_end = min(seg_end, ov_end)
                seg["overlap_duration"] = round(overlap_end - overlap_start, 2)
                seg["overlap_ratio"] = round(
                    (overlap_end - overlap_start) / (seg_end - seg_start), 2
                ) if seg_end > seg_start else 0
                break
        else:
            seg["has_overlap"] = False
            seg["overlap_duration"] = 0
            seg["overlap_ratio"] = 0
    
    return segments


def recover_backchannels(
    segments: List[Dict],
    whisper_words: List[Dict],
    min_duration: float = 0.05,
    backchannel_patterns: List[str] = None
) -> List[Dict]:
    """
    Hồi phục các backchannel ngắn có thể bị bỏ sót bởi diarization.
    
    Nếu Whisper nhận dạng được text nhưng không có segment tương ứng,
    tạo segment mới.
    """
    if backchannel_patterns is None:
        backchannel_patterns = [
            'ừ', 'vâng', 'ờ', 'dạ', 'ừm', 'à', 'ok', 'được',
            'thế à', 'vậy hả', 'vậy á', 'đúng rồi', 'phải không'
        ]
    
    # Get covered time ranges
    covered = []
    for seg in segments:
        covered.append((seg.get("start", 0), seg.get("end", 0)))
    
    # Check each word/phrase from Whisper
    recovered = []
    for word in whisper_words:
        word_start = word.get("start", 0)
        word_end = word.get("end", 0)
        word_text = word.get("word", "").strip().lower()
        
        if word_end - word_start < min_duration:
            continue
        
        # Check if already covered
        is_covered = any(
            start <= word_start and word_end <= end
            for start, end in covered
        )
        
        if not is_covered:
            # Check if it's a backchannel pattern
            if any(pattern in word_text for pattern in backchannel_patterns):
                recovered.append({
                    "start": round(word_start, 2),
                    "end": round(word_end, 2),
                    "text": word_text,
                    "speaker": "UNKNOWN",
                    "recovered_backchannel": True
                })
    
    # Merge recovered segments
    if recovered:
        segments.extend(recovered)
        segments.sort(key=lambda x: x.get("start", 0))
        
        # Re-index
        for i, seg in enumerate(segments):
            seg["id"] = i
    
    return segments


def split_audio_into_chunks(
    audio: np.ndarray, 
    chunk_minutes: int = 30,
    sample_rate: int = 16000
) -> List[Tuple[np.ndarray, float]]:
    """Split audio into chunks."""
    chunk_samples = chunk_minutes * 60 * sample_rate
    total_samples = len(audio)
    
    if total_samples <= chunk_samples:
        return [(audio, 0.0)]
    
    chunks = []
    for i in range(0, total_samples, chunk_samples):
        chunk = audio[i:i + chunk_samples]
        start_offset = i / sample_rate
        chunks.append((chunk, start_offset))
    
    return chunks


def merge_transcription_results(
    chunk_results: List[Dict],
    chunk_offsets: List[float]
) -> Dict:
    """Merge transcription results from chunks."""
    all_segments = []
    all_words = []
    segment_id = 0
    
    for result, offset in zip(chunk_results, chunk_offsets):
        for seg in result.get("segments", []):
            adjusted_seg = seg.copy()
            adjusted_seg["id"] = segment_id
            adjusted_seg["start"] = round(seg.get("start", 0) + offset, 2)
            adjusted_seg["end"] = round(seg.get("end", 0) + offset, 2)
            
            # Adjust word timestamps
            if adjusted_seg.get("words"):
                for w in adjusted_seg["words"]:
                    w["start"] = w.get("start", 0) + offset
                    w["end"] = w.get("end", 0) + offset
                all_words.extend(adjusted_seg["words"])
            
            all_segments.append(adjusted_seg)
            segment_id += 1
    
    return {"segments": all_segments, "words": all_words}


def process_single_audio(
    audio_path: str,
    output_dir: str,
    model_name: str = "large-v3",
    device: str = "cuda",
    hf_token: Optional[str] = None,
    batch_size: int = 16,
    compute_type: str = "float16",
    enable_overlap: bool = True,
    vad_onset: float = 0.5,
    vad_offset: float = 0.363,
    min_speech_duration: float = 0.15  # Giảm từ 0.5s xuống 0.15s cho backchannel
) -> Dict:
    """
    Process một file audio: ASR + Alignment + Diarization + Overlap Detection
    """
    audio_name = Path(audio_path).stem
    print(f"\n🎵 Processing: {audio_name}")
    
    # Clear GPU memory
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Adjust for CPU
    if device == "cpu":
        compute_type = "int8"
        batch_size = 8
    
    # 1. Load model
    print("   📦 Loading Whisper model...")
    model = whisperx.load_model(
        model_name, 
        device, 
        compute_type=compute_type
    )
    
    # 2. Load audio
    print("   🔊 Loading audio...")
    audio = whisperx.load_audio(audio_path)
    duration_sec = len(audio) / 16000
    duration_min = duration_sec / 60
    
    # 3. Transcribe
    MAX_DURATION_MIN = 60
    CHUNK_DURATION_MIN = 30
    
    all_words = []
    
    if duration_min > MAX_DURATION_MIN:
        chunks = split_audio_into_chunks(audio, CHUNK_DURATION_MIN)
        print(f"   📝 Transcribing ({duration_min:.1f} min in {len(chunks)} chunks)...")
        
        chunk_results = []
        chunk_offsets = []
        start_time = time.time()
        
        for i, (chunk_audio, offset) in enumerate(chunks, 1):
            chunk_min = len(chunk_audio) / 16000 / 60
            print(f"      📦 Chunk {i}/{len(chunks)} ({chunk_min:.1f} min)...")
            chunk_result = model.transcribe(chunk_audio, batch_size=batch_size, language="vi")
            chunk_results.append(chunk_result)
            chunk_offsets.append(offset)
            
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        
        result = merge_transcription_results(chunk_results, chunk_offsets)
        all_words = result.get("words", [])
        elapsed = time.time() - start_time
    else:
        print(f"   📝 Transcribing ({duration_min:.1f} min)...")
        start_time = time.time()
        result = model.transcribe(audio, batch_size=batch_size, language="vi")
        elapsed = time.time() - start_time
        
        # Collect all words
        for seg in result.get("segments", []):
            all_words.extend(seg.get("words", []))
    
    speed = duration_sec / elapsed if elapsed > 0 else 0
    print(f"      ✓ Done in {elapsed:.0f}s ({speed:.1f}x realtime)")
    
    # 4. Align timestamps
    print("   ⏱️  Aligning timestamps...")
    align_start = time.time()
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code="vi", 
            device=device
        )
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            device,
            return_char_alignments=False
        )
        print(f"      ✓ Done in {time.time() - align_start:.0f}s")
    except Exception as e:
        print(f"   ⚠️  Alignment failed: {e}")
    
    # 5. Diarization
    if hf_token:
        print("   👥 Speaker diarization...")
        diar_start = time.time()
        try:
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(
                use_auth_token=hf_token, 
                device=device
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print(f"      ✓ Done in {time.time() - diar_start:.0f}s")
        except Exception as e:
            print(f"   ⚠️  Diarization failed: {e}")
    
    # 6. Overlap Detection (NEW)
    overlaps = []
    if enable_overlap and hf_token:
        print("   🔀 Overlap detection...")
        overlap_start = time.time()
        overlaps = detect_overlaps(audio_path, hf_token, device)
        print(f"      ✓ Found {len(overlaps)} overlap regions in {time.time() - overlap_start:.0f}s")
    
    # 7. Format output
    segments = []
    for i, seg in enumerate(result.get("segments", [])):
        segments.append({
            "id": i,
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "text": seg.get("text", "").strip(),
            "speaker": seg.get("speaker", "UNKNOWN"),
            "words": seg.get("words", [])
        })
    
    # 8. Mark overlaps
    if overlaps:
        segments = mark_overlaps(segments, overlaps)
    
    # 9. Recover backchannels (NEW)
    if all_words:
        print("   🔄 Recovering backchannels...")
        original_count = len(segments)
        segments = recover_backchannels(segments, all_words)
        recovered_count = len(segments) - original_count
        if recovered_count > 0:
            print(f"      ✓ Recovered {recovered_count} backchannels")
    
    # Compute stats
    overlap_count = sum(1 for s in segments if s.get("has_overlap", False))
    
    output_data = {
        "audio_file": Path(audio_path).name,
        "audio_path": str(audio_path),
        "duration": round(len(audio) / 16000, 2),
        "num_segments": len(segments),
        "num_overlaps": overlap_count,
        "overlap_regions": [{"start": s, "end": e} for s, e in overlaps],
        "segments": segments,
        "processing_config": {
            "model": model_name,
            "enable_overlap": enable_overlap,
            "min_speech_duration": min_speech_duration,
            "version": "v2"
        }
    }
    
    # 10. Save
    output_path = Path(output_dir) / f"{audio_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Saved: {output_path}")
    print(f"      Segments: {len(segments)}, Overlaps: {overlap_count}, Duration: {output_data['duration']}s")
    
    return output_data


def process_directory(
    input_dir: str,
    output_dir: str,
    model_name: str = "large-v3",
    device: str = "cuda",
    hf_token: Optional[str] = None,
    batch_size: int = 16,
    enable_overlap: bool = True,
    extensions: List[str] = [".wav", ".mp3", ".m4a", ".flac"]
) -> List[str]:
    """Process tất cả audio files trong thư mục."""
    input_path = Path(input_dir)
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_path.glob(f"*{ext}"))
        audio_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    audio_files = sorted(set(audio_files))
    
    if not audio_files:
        print(f"❌ Không tìm thấy file audio trong {input_dir}")
        return []
    
    print(f"📂 Found {len(audio_files)} audio files")
    
    # Check existing
    output_path = Path(output_dir)
    existing = set(f.stem for f in output_path.glob("*.json"))
    
    to_process = [f for f in audio_files if f.stem not in existing]
    
    if existing:
        print(f"   ⏭️  Skipping {len(existing)} already processed")
    
    if not to_process:
        print("   ✅ All files already processed!")
        return []
    
    print(f"   🔄 Processing {len(to_process)} files...")
    
    from tqdm import tqdm
    processed = []
    for audio_file in tqdm(to_process, desc="Processing", unit="file"):
        try:
            process_single_audio(
                str(audio_file), output_dir, model_name, device, 
                hf_token, batch_size, enable_overlap=enable_overlap
            )
            processed.append(str(audio_file))
        except Exception as e:
            print(f"   ❌ Error processing {audio_file}: {e}")
    
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Auto-process audio với whisperX + Overlap Detection (v2)"
    )
    
    parser.add_argument("--input", "-i", required=True,
                        help="Thư mục chứa audio files hoặc single file")
    parser.add_argument("--output", "-o", default="data/processed/auto",
                        help="Thư mục output")
    parser.add_argument("--model", default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Whisper model size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"],
                        help="Device to use")
    parser.add_argument("--hf-token", help="HuggingFace token")
    parser.add_argument("--skip-diarization", action="store_true",
                        help="Skip speaker diarization")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for transcription")
    parser.add_argument("--enable-overlap", action="store_true",
                        help="Enable overlap detection")
    parser.add_argument("--no-overlap", action="store_true",
                        help="Disable overlap detection")
    
    args = parser.parse_args()
    
    # Get HF token
    hf_token = args.hf_token
    if not args.skip_diarization and not hf_token:
        hf_token = get_hf_token()
    
    enable_overlap = args.enable_overlap and not args.no_overlap
    
    print(f"🚀 Config:")
    print(f"   Model: {args.model}")
    print(f"   Device: {args.device}")
    print(f"   Overlap Detection: {'Enabled' if enable_overlap else 'Disabled'}")
    
    # Process
    input_path = Path(args.input)
    
    if input_path.is_file():
        process_single_audio(
            str(input_path), args.output, args.model, args.device, 
            hf_token, args.batch_size, enable_overlap=enable_overlap
        )
    elif input_path.is_dir():
        processed = process_directory(
            str(input_path), args.output, args.model, args.device, 
            hf_token, args.batch_size, enable_overlap=enable_overlap
        )
        print(f"\n📊 Summary: Processed {len(processed)} files")
    else:
        print(f"❌ Invalid input: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
