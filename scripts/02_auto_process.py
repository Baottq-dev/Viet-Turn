#!/usr/bin/env python3
"""
Script 02: Auto-process audio với whisperX (ASR + Diarization)

Yêu cầu:
    pip install whisperx torch torchaudio
    HuggingFace token cho pyannote: https://huggingface.co/settings/tokens

Usage:
    python scripts/02_auto_process.py --input data/raw --output data/processed/auto
    python scripts/02_auto_process.py --input data/raw --output data/processed/auto --device cuda
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import warnings
import numpy as np

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

warnings.filterwarnings("ignore")

# Check imports
try:
    import torch
    import whisperx
    
    # Fix PyTorch 2.6+ weights_only issue with pyannote models
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
    """Lấy HuggingFace token từ env hoặc hỏi user"""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if not token:
        print("⚠️  Cần HuggingFace token cho speaker diarization")
        print("   Lấy token tại: https://huggingface.co/settings/tokens")
        print("   Sau đó: export HF_TOKEN='your_token'")
        token = input("   Nhập token (hoặc Enter để skip diarization): ").strip()
    
    return token


def split_audio_into_chunks(
    audio: np.ndarray, 
    chunk_minutes: int = 30,
    sample_rate: int = 16000
) -> List[tuple]:
    """
    Split audio into chunks of specified duration.
    
    Returns:
        List of (audio_chunk, start_time_offset) tuples
    """
    chunk_samples = chunk_minutes * 60 * sample_rate
    total_samples = len(audio)
    
    if total_samples <= chunk_samples:
        return [(audio, 0.0)]
    
    chunks = []
    for i in range(0, total_samples, chunk_samples):
        chunk = audio[i:i + chunk_samples]
        start_offset = i / sample_rate  # seconds
        chunks.append((chunk, start_offset))
    
    return chunks


def merge_transcription_results(
    chunk_results: List[Dict],
    chunk_offsets: List[float]
) -> Dict:
    """
    Merge transcription results from multiple chunks.
    Adjusts timestamps based on chunk offsets.
    """
    all_segments = []
    segment_id = 0
    
    for result, offset in zip(chunk_results, chunk_offsets):
        for seg in result.get("segments", []):
            adjusted_seg = seg.copy()
            adjusted_seg["id"] = segment_id
            adjusted_seg["start"] = round(seg.get("start", 0) + offset, 2)
            adjusted_seg["end"] = round(seg.get("end", 0) + offset, 2)
            
            # Adjust word timestamps if present
            if adjusted_seg.get("words"):
                adjusted_seg["words"] = [
                    {**w, "start": w.get("start", 0) + offset, "end": w.get("end", 0) + offset}
                    for w in adjusted_seg["words"]
                ]
            
            all_segments.append(adjusted_seg)
            segment_id += 1
    
    return {"segments": all_segments}


def process_single_audio(
    audio_path: str,
    output_dir: str,
    model_name: str = "large-v3",
    device: str = "cuda",
    hf_token: Optional[str] = None,
    batch_size: int = 16,
    compute_type: str = "float16"
) -> Dict:
    """
    Process một file audio: ASR + Alignment + Diarization
    
    Returns:
        Dict chứa segments với speaker + text + timestamps
    """
    audio_name = Path(audio_path).stem
    print(f"\n🎵 Processing: {audio_name}")
    
    # Clear GPU memory before processing
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
    
    # 3. Transcribe (with chunking for long files)
    MAX_DURATION_MIN = 60  # Chunk files longer than 60 min
    CHUNK_DURATION_MIN = 30
    
    if duration_min > MAX_DURATION_MIN:
        # Split into chunks
        chunks = split_audio_into_chunks(audio, CHUNK_DURATION_MIN)
        print(f"   📝 Transcribing ({duration_min:.1f} min audio in {len(chunks)} chunks)...")
        
        chunk_results = []
        chunk_offsets = []
        start_time = time.time()
        
        for i, (chunk_audio, offset) in enumerate(chunks, 1):
            chunk_min = len(chunk_audio) / 16000 / 60
            print(f"      📦 Chunk {i}/{len(chunks)} ({chunk_min:.1f} min)...")
            chunk_result = model.transcribe(chunk_audio, batch_size=batch_size, language="vi")
            chunk_results.append(chunk_result)
            chunk_offsets.append(offset)
            
            # Clear GPU memory between chunks
            import gc
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        
        # Merge results
        result = merge_transcription_results(chunk_results, chunk_offsets)
        elapsed = time.time() - start_time
    else:
        print(f"   📝 Transcribing ({duration_min:.1f} min audio)...")
        start_time = time.time()
        result = model.transcribe(audio, batch_size=batch_size, language="vi")
        elapsed = time.time() - start_time
    
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
    
    # 5. Diarization (nếu có token)
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
    
    # 6. Format output
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
    
    output_data = {
        "audio_file": Path(audio_path).name,
        "audio_path": str(audio_path),
        "duration": round(len(audio) / 16000, 2),  # seconds
        "num_segments": len(segments),
        "segments": segments
    }
    
    # 7. Save
    output_path = Path(output_dir) / f"{audio_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Saved: {output_path}")
    print(f"      Segments: {len(segments)}, Duration: {output_data['duration']}s")
    
    return output_data


def process_directory(
    input_dir: str,
    output_dir: str,
    model_name: str = "large-v3",
    device: str = "cuda",
    hf_token: Optional[str] = None,
    batch_size: int = 16,
    extensions: List[str] = [".wav", ".mp3", ".m4a", ".flac"]
) -> List[str]:
    """
    Process tất cả audio files trong thư mục.
    
    Returns:
        List các file đã xử lý thành công
    """
    input_path = Path(input_dir)
    
    # Find audio files
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
                str(audio_file), output_dir, model_name, device, hf_token, batch_size
            )
            processed.append(str(audio_file))
        except Exception as e:
            print(f"   ❌ Error processing {audio_file}: {e}")
    
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Auto-process audio với whisperX (ASR + Diarization)"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Thư mục chứa audio files hoặc single file"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/auto",
        help="Thư mục output (default: data/processed/auto)"
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: large-v3)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda if available)"
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token cho diarization (hoặc set HF_TOKEN env)"
    )
    parser.add_argument(
        "--skip-diarization",
        action="store_true",
        help="Skip speaker diarization"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for transcription (reduce if CUDA OOM, default: 16)"
    )
    
    args = parser.parse_args()
    
    # Get HF token
    hf_token = args.hf_token
    if not args.skip_diarization and not hf_token:
        hf_token = get_hf_token()
    
    # Process
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        process_single_audio(
            str(input_path), args.output, args.model, args.device, hf_token, args.batch_size
        )
    elif input_path.is_dir():
        # Directory
        processed = process_directory(
            str(input_path), args.output, args.model, args.device, hf_token, args.batch_size
        )
        print(f"\n📊 Summary: Processed {len(processed)} files")
    else:
        print(f"❌ Invalid input: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
