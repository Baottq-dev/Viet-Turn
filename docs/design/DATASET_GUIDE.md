# Dataset Construction Guide for MM-VAP-VI

## Chi tiết cách xây dựng bộ dataset mới cho kiến trúc Multimodal VAP

---

## 1. So sánh pipeline cũ vs mới

### 1.1 Pipeline hiện tại (Weakly-supervised, segment-level)

```
Audio (mono)
  │
  ▼
faster-whisper (ASR) → transcript có word-level timestamps
  │
  ▼
Gemini 2.0 Flash (LLM) → phân tích transcript:
  - Gán speaker cho từng đoạn (diarization bằng LLM)
  - Chia thành từng lượt nói (turn segmentation)
  - Gán label: YIELD / HOLD / BACKCHANNEL / COOPERATIVE_INTERRUPT / COMPETITIVE_INTERRUPT
  - Trả về JSON với text + label + speaker + confidence
  │
  ▼
05_prepare_dataset.py → merge 5-class → 3-class, fix timestamps, validate
  │
  ▼
06_split_dataset.py → train/val/test split (by file)
  │
  ▼
Output: train.json = [{id, start, end, text, speaker, label, audio_file}, ...]
        MỖI SAMPLE = 1 LƯỢT NÓI HOÀN CHỈNH + 1 LABEL
```

**Vấn đề với pipeline hiện tại cho VAP:**

| Vấn đề | Giải thích |
|---------|-----------|
| Label ở segment-level | VAP cần label ở frame-level (mỗi 20ms) |
| Diarization bằng LLM | LLM diarize từ text, không từ audio → không đáng tin cậy cho voice activity detection |
| Timestamps gián tiếp | Timestamps đến từ ASR word alignment, không phải voice activity timestamps |
| Không có voice activity matrix | VAP cần biết chính xác frame nào speaker nào đang nói |
| Chỉ gán nhãn tại turn boundaries | VAP cần nhãn liên tục tại MỌI frame |

### 1.2 Pipeline mới (Self-supervised, frame-level)

```
Audio (mono, 16kHz)
  │
  ├──► pyannote/speaker-diarization-3.1 → per-speaker voice activity
  │      Output: [(speaker_id, start_sec, end_sec), ...]
  │
  ├──► PhoWhisper (streaming ASR) → word-level transcript + timestamps
  │      Output: [{word, start, end}, ...]
  │
  ▼
VAP Label Generator → tại MỖI FRAME (20ms):
  nhìn 2s tương lai → voice activity của 2 speakers → 256-class label
  │
  ▼
Output: per file:
  audio.wav                    (raw audio)
  voice_activity.npy           (2, num_frames) — binary matrix
  vap_labels.npy               (num_frames,) — class index [0-255]
  transcript.json              (word-level timestamps)
  metadata.json                (speakers, duration, source)
```

**Ưu điểm của pipeline mới:**

| Ưu điểm | Giải thích |
|---------|-----------|
| Self-supervised labels | Ground truth tự động từ actual voice activity — không cần LLM gán nhãn |
| Frame-level resolution | 1 label mỗi 20ms → ~15,000 labels/phút thay vì ~10 labels/phút |
| Diarization chính xác | pyannote 3.1 DER ~10-15% trên Vietnamese (tốt hơn nhiều so với LLM) |
| Scalable | Chỉ cần audio → chạy pipeline → done. Không tốn tiền LLM API |
| Reproducible | Deterministic pipeline, không phụ thuộc vào LLM output stochastic |

---

## 2. Data Collection

### 2.1 Nguồn dữ liệu

```
┌─────────────────────────────────────────────────────────────┐
│  NGUỒN DỮ LIỆU (by priority)                                │
│                                                              │
│  1. YouTube Podcasts tiếng Việt                              │
│     - 2 người nói, đối thoại tự nhiên                        │
│     - Ví dụ: kênh phỏng vấn, talk show, thảo luận           │
│     - Chất lượng audio: thường tốt (studio recording)        │
│     - Ước tính: 50-100 videos × 30-60 phút = 25-100 giờ     │
│                                                              │
│  2. Podcast platforms (Spotify, Apple Podcasts)              │
│     - RSS feed → download trực tiếp                          │
│     - Nhiều thể loại: chính trị, giải trí, giáo dục         │
│                                                              │
│  3. Radio talk shows tiếng Việt                              │
│     - Call-in shows → có cả kênh điện thoại                  │
│     - Đa dạng phương ngữ                                     │
│                                                              │
│  4. Public meeting recordings                                │
│     - Hội nghị, seminar có Q&A                               │
│     - Nhiều hơn 2 speakers → cần lọc                         │
│                                                              │
│  LƯU Ý VỀ BẢN QUYỀN:                                       │
│  - Chỉ dùng cho research (fair use)                          │
│  - Ghi rõ nguồn trong paper                                  │
│  - Không distribute raw audio                                │
│  - Chỉ share processed features nếu cần                      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Yêu cầu kỹ thuật

```
Audio format:
  - Sample rate: 16kHz (resample nếu cần)
  - Channels: Mono (convert nếu stereo)
  - Bit depth: 16-bit PCM hoặc float32
  - Format: WAV (preferred) hoặc FLAC
  - Noise level: SNR > 15dB (loại bỏ nếu quá ồn)

Content requirements:
  - Đúng 2 speakers chính (có thể có thêm phụ nhưng filter)
  - Đối thoại tự nhiên (không đọc script)
  - Có turn transitions thường xuyên (> 5 transitions/phút)
  - Thời lượng mỗi file: 10-60 phút

Diversity:
  - Giới tính: nam-nam, nữ-nữ, nam-nữ (cân bằng)
  - Phương ngữ: Bắc, Trung, Nam (nếu có thể)
  - Chủ đề: đa dạng (chính trị, giải trí, kỹ thuật, đời sống)
  - Phong cách: formal (phỏng vấn) + casual (chat)
```

### 2.3 Download & Preprocessing

```python
# scripts/00_download_audio.py (pseudocode)

# Step 1: Download từ YouTube
# Dùng yt-dlp (không phải youtube-dl, đã deprecated)
# yt-dlp -x --audio-format wav --audio-quality 0 -o "%(title)s.%(ext)s" <URL>

# Step 2: Convert to 16kHz mono WAV
import librosa
import soundfile as sf

def preprocess_audio(input_path, output_path, target_sr=16000):
    """Convert audio to 16kHz mono WAV."""
    audio, sr = librosa.load(input_path, sr=target_sr, mono=True)

    # Optional: noise reduction
    # import noisereduce as nr
    # audio = nr.reduce_noise(y=audio, sr=sr)

    # Normalize volume
    audio = audio / max(abs(audio).max(), 1e-6)

    sf.write(output_path, audio, target_sr)
    return len(audio) / target_sr  # duration in seconds
```

---

## 3. Speaker Diarization

### 3.1 Tool: pyannote/speaker-diarization-3.1

Đây là diarization model SOTA hiện tại. Diarization Error Rate (DER) ~10-13% trên các benchmark.

```python
# scripts/01_diarize.py

from pyannote.audio import Pipeline
import torch
import json
from pathlib import Path

def diarize_audio(audio_path, hf_token, num_speakers=2):
    """
    Run speaker diarization on audio file.

    Returns:
        List[Dict] with keys: speaker, start, end
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    # GPU nếu có
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    # Run diarization
    diarization = pipeline(
        audio_path,
        num_speakers=num_speakers  # biết trước số speakers
    )

    # Convert to list of segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3)
        })

    return segments
```

### 3.2 Voice Activity Matrix Generation

```python
# scripts/02_voice_activity.py

import numpy as np
from typing import List, Dict

FRAME_RATE = 50  # 50fps = 20ms per frame (VAP standard)

def build_voice_activity_matrix(
    diarization_segments: List[Dict],
    total_duration_sec: float,
    frame_rate: int = FRAME_RATE,
    num_speakers: int = 2
) -> np.ndarray:
    """
    Build binary voice activity matrix from diarization output.

    Args:
        diarization_segments: List of {speaker, start, end}
        total_duration_sec: Total audio duration in seconds
        frame_rate: Frames per second (default 50 = 20ms)
        num_speakers: Number of speakers (default 2)

    Returns:
        va_matrix: np.ndarray, shape (num_speakers, total_frames), dtype=float32
        Values: 1.0 = speaking, 0.0 = silent

    Example:
        Speaker 1 nói từ 0.0-2.5s, Speaker 2 nói từ 2.7-4.1s
        Duration = 5.0s → 250 frames

        va_matrix[0, 0:125] = 1.0    # SP1 active frames 0-124
        va_matrix[0, 125:250] = 0.0  # SP1 silent frames 125-249
        va_matrix[1, 0:135] = 0.0    # SP2 silent frames 0-134
        va_matrix[1, 135:205] = 1.0  # SP2 active frames 135-204
        va_matrix[1, 205:250] = 0.0  # SP2 silent frames 205-249
    """
    total_frames = int(total_duration_sec * frame_rate)
    va_matrix = np.zeros((num_speakers, total_frames), dtype=np.float32)

    # Map speaker labels to indices
    # pyannote output có labels như "SPEAKER_00", "SPEAKER_01"
    speaker_labels = sorted(set(seg["speaker"] for seg in diarization_segments))

    if len(speaker_labels) > num_speakers:
        # Nếu có > 2 speakers, giữ 2 speaker nói nhiều nhất
        speaker_durations = {}
        for seg in diarization_segments:
            spk = seg["speaker"]
            dur = seg["end"] - seg["start"]
            speaker_durations[spk] = speaker_durations.get(spk, 0) + dur

        speaker_labels = sorted(
            speaker_durations.keys(),
            key=lambda s: speaker_durations[s],
            reverse=True
        )[:num_speakers]

    speaker_to_idx = {spk: i for i, spk in enumerate(speaker_labels)}

    # Fill voice activity matrix
    for seg in diarization_segments:
        spk = seg["speaker"]
        if spk not in speaker_to_idx:
            continue  # skip minor speakers

        idx = speaker_to_idx[spk]
        start_frame = int(seg["start"] * frame_rate)
        end_frame = int(seg["end"] * frame_rate)

        # Clamp to valid range
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(0, min(end_frame, total_frames))

        va_matrix[idx, start_frame:end_frame] = 1.0

    return va_matrix, speaker_to_idx
```

---

## 4. VAP Label Generation

### 4.1 Core Algorithm

```python
# scripts/03_generate_labels.py

import numpy as np

# VAP bin boundaries (milliseconds from current frame)
# Bins have increasing duration: near-future needs precision, far-future less so
VAP_BINS_MS = [
    (0, 200),      # Bin 0: 0-200ms    (10 frames) — immediate
    (200, 600),     # Bin 1: 200-600ms  (20 frames) — short-term
    (600, 1200),    # Bin 2: 600-1200ms (30 frames) — medium-term
    (1200, 2000),   # Bin 3: 1200-2000ms(40 frames) — long-term
]

# Tại sao bins không đều?
# - Bin 0 (200ms): Cần precision cao để detect immediate turn shifts.
#   Con người phản ứng trong ~200ms.
# - Bin 3 (800ms): Chỉ cần rough prediction. Xa hơn 1.2s
#   thì prediction accuracy giảm mạnh anyway.
# - Đây là thiết kế từ VAP gốc (Ekstedt & Skantze, 2022).

def generate_vap_labels(
    va_matrix: np.ndarray,
    frame_rate: int = 50,
    bins_ms: list = None,
    activity_threshold: float = 0.5
) -> np.ndarray:
    """
    Generate VAP 256-class labels from voice activity matrix.

    Args:
        va_matrix: (2, T) binary voice activity matrix
        frame_rate: frames per second
        bins_ms: list of (start_ms, end_ms) tuples for projection bins
        activity_threshold: ratio threshold for bin activity

    Returns:
        labels: (T,) int64 array, values in [0, 255]

    Algorithm:
        For each frame t:
            For each of 4 bins:
                For each of 2 speakers:
                    ratio = mean(va_matrix[speaker, bin_start:bin_end])
                    bit = 1 if ratio > threshold else 0
            Encode 8 bits as integer [0-255]
    """
    if bins_ms is None:
        bins_ms = VAP_BINS_MS

    num_speakers, total_frames = va_matrix.shape
    assert num_speakers == 2, "VAP requires exactly 2 speakers"

    labels = np.zeros(total_frames, dtype=np.int64)

    # Pre-compute bin boundaries in frames
    bins_frames = []
    for start_ms, end_ms in bins_ms:
        start_f = int(start_ms / 1000 * frame_rate)
        end_f = int(end_ms / 1000 * frame_rate)
        bins_frames.append((start_f, end_f))

    for t in range(total_frames):
        bits = []

        for speaker in range(num_speakers):
            for bin_start, bin_end in bins_frames:
                # Look ahead from current frame
                abs_start = t + bin_start
                abs_end = t + bin_end

                # Clamp to valid range
                abs_start = min(abs_start, total_frames)
                abs_end = min(abs_end, total_frames)

                if abs_start >= abs_end:
                    # Beyond audio end → assume silence
                    bits.append(0)
                else:
                    # Compute activity ratio in this bin
                    ratio = va_matrix[speaker, abs_start:abs_end].mean()
                    bits.append(1 if ratio > activity_threshold else 0)

        # Encode 8 bits as integer
        # Bit order: [sp1_bin0, sp1_bin1, sp1_bin2, sp1_bin3,
        #             sp2_bin0, sp2_bin1, sp2_bin2, sp2_bin3]
        class_index = 0
        for i, bit in enumerate(bits):
            class_index += bit * (2 ** i)

        labels[t] = class_index

    return labels


def generate_vap_labels_vectorized(
    va_matrix: np.ndarray,
    frame_rate: int = 50,
    bins_ms: list = None,
    activity_threshold: float = 0.5
) -> np.ndarray:
    """
    Vectorized version (much faster for large files).

    Same interface as generate_vap_labels but uses cumulative sums
    for efficient bin activity computation.
    """
    if bins_ms is None:
        bins_ms = VAP_BINS_MS

    num_speakers, total_frames = va_matrix.shape
    assert num_speakers == 2

    # Pre-compute cumulative sums for efficient window means
    # cumsum[i] = sum(va[0:i])
    # mean(va[a:b]) = (cumsum[b] - cumsum[a]) / (b - a)
    cumsum = np.zeros((num_speakers, total_frames + 1), dtype=np.float64)
    for s in range(num_speakers):
        cumsum[s, 1:] = np.cumsum(va_matrix[s])

    bins_frames = []
    for start_ms, end_ms in bins_ms:
        start_f = int(start_ms / 1000 * frame_rate)
        end_f = int(end_ms / 1000 * frame_rate)
        bins_frames.append((start_f, end_f))

    # Compute all bits at once
    # bits_matrix: (num_speakers * num_bins, total_frames)
    bits_list = []

    for speaker in range(num_speakers):
        for bin_start, bin_end in bins_frames:
            # For each frame t, compute mean activity in [t+bin_start, t+bin_end]
            abs_starts = np.arange(total_frames) + bin_start
            abs_ends = np.arange(total_frames) + bin_end

            # Clamp
            abs_starts = np.clip(abs_starts, 0, total_frames)
            abs_ends = np.clip(abs_ends, 0, total_frames)

            # Avoid division by zero
            lengths = abs_ends - abs_starts
            lengths = np.maximum(lengths, 1)

            # Compute means using cumulative sums
            sums = cumsum[speaker, abs_ends] - cumsum[speaker, abs_starts]
            ratios = sums / lengths

            bits = (ratios > activity_threshold).astype(np.int64)
            bits_list.append(bits)

    # Stack: (8, T)
    bits_matrix = np.stack(bits_list, axis=0)

    # Encode as class indices
    powers = np.array([2**i for i in range(len(bits_list))], dtype=np.int64)
    labels = (bits_matrix.T @ powers).astype(np.int64)  # (T,)

    return labels
```

### 4.2 Label Interpretation

```python
# scripts/utils/vap_utils.py

import numpy as np
from typing import Dict, List

def decode_vap_label(class_index: int, num_bins: int = 4) -> Dict:
    """
    Decode a VAP class index back to speaker activity pattern.

    Args:
        class_index: integer in [0, 255]
        num_bins: number of time bins per speaker (default 4)

    Returns:
        Dict with keys 'speaker_1' and 'speaker_2', each a list of 0/1
    """
    bits = []
    val = class_index
    for _ in range(num_bins * 2):
        bits.append(val % 2)
        val //= 2

    return {
        "speaker_1": bits[:num_bins],        # [bin0, bin1, bin2, bin3]
        "speaker_2": bits[num_bins:],        # [bin0, bin1, bin2, bin3]
    }


# Pre-defined class groups for event evaluation
# These mappings allow extracting turn-taking events from VAP predictions

def get_shift_class_indices(num_bins=4):
    """
    Get class indices where Speaker 1 stops and Speaker 2 starts.

    Pattern: SP1 active in early bins, inactive in later bins
             SP2 inactive in early bins, active in later bins

    Examples:
        SP1=[1,1,0,0] SP2=[0,0,1,1] → shift happening now
        SP1=[1,0,0,0] SP2=[0,0,1,1] → shift imminent
        SP1=[1,1,0,0] SP2=[0,1,1,1] → shift with slight overlap
    """
    shift_indices = []
    for i in range(256):
        decoded = decode_vap_label(i, num_bins)
        sp1 = decoded["speaker_1"]
        sp2 = decoded["speaker_2"]

        # SP1 starts active, becomes inactive
        # SP2 starts inactive, becomes active
        sp1_fading = (sum(sp1[:2]) > 0 and sum(sp1[2:]) == 0)
        sp2_rising = (sum(sp2[:2]) == 0 and sum(sp2[2:]) > 0)

        # Also include cases where SP1 is already gone
        sp1_gone = (sum(sp1) == 0)
        sp2_coming = (sum(sp2[1:]) > 0)

        if (sp1_fading and sp2_rising) or (sp1_gone and sp2_coming):
            shift_indices.append(i)

    # Also add reverse direction (SP2 → SP1)
    for i in range(256):
        decoded = decode_vap_label(i, num_bins)
        sp1 = decoded["speaker_1"]
        sp2 = decoded["speaker_2"]

        sp2_fading = (sum(sp2[:2]) > 0 and sum(sp2[2:]) == 0)
        sp1_rising = (sum(sp1[:2]) == 0 and sum(sp1[2:]) > 0)

        sp2_gone = (sum(sp2) == 0)
        sp1_coming = (sum(sp1[1:]) > 0)

        if (sp2_fading and sp1_rising) or (sp2_gone and sp1_coming):
            if i not in shift_indices:
                shift_indices.append(i)

    return shift_indices


def get_hold_class_indices(num_bins=4):
    """
    Get class indices where current speaker continues (hold).

    Pattern: One speaker active throughout, other silent throughout.
    """
    hold_indices = []
    for i in range(256):
        decoded = decode_vap_label(i, num_bins)
        sp1 = decoded["speaker_1"]
        sp2 = decoded["speaker_2"]

        # SP1 holds (all bins active, SP2 all silent)
        if sum(sp1) == num_bins and sum(sp2) == 0:
            hold_indices.append(i)

        # SP2 holds
        if sum(sp2) == num_bins and sum(sp1) == 0:
            hold_indices.append(i)

    return hold_indices


def get_backchannel_class_indices(num_bins=4):
    """
    Get class indices where one speaker gives a short backchannel
    while the other maintains the floor.

    Pattern: Primary speaker active in most bins,
             secondary speaker active in only 1-2 bins (short response).
    """
    bc_indices = []
    for i in range(256):
        decoded = decode_vap_label(i, num_bins)
        sp1 = decoded["speaker_1"]
        sp2 = decoded["speaker_2"]

        # SP1 maintains floor, SP2 gives backchannel
        if sum(sp1) >= 3 and 1 <= sum(sp2) <= 2:
            bc_indices.append(i)

        # SP2 maintains floor, SP1 gives backchannel
        if sum(sp2) >= 3 and 1 <= sum(sp1) <= 2:
            if i not in bc_indices:
                bc_indices.append(i)

    return bc_indices
```

### 4.3 Label Statistics & Validation

```python
# scripts/utils/label_stats.py

import numpy as np
from collections import Counter

def analyze_vap_labels(labels: np.ndarray, frame_rate: int = 50):
    """
    Analyze VAP label distribution and turn-taking statistics.
    """
    total_frames = len(labels)
    total_duration = total_frames / frame_rate

    # Class distribution
    counts = Counter(labels.tolist())

    # Event counts
    shift_indices = set(get_shift_class_indices())
    hold_indices = set(get_hold_class_indices())
    bc_indices = set(get_backchannel_class_indices())
    silence_index = 0  # all zeros = both silent

    shift_frames = sum(1 for l in labels if l in shift_indices)
    hold_frames = sum(1 for l in labels if l in hold_indices)
    bc_frames = sum(1 for l in labels if l in bc_indices)
    silence_frames = sum(1 for l in labels if l == silence_index)
    other_frames = total_frames - shift_frames - hold_frames - bc_frames - silence_frames

    stats = {
        "total_frames": total_frames,
        "total_duration_sec": round(total_duration, 1),
        "unique_classes": len(counts),
        "top_10_classes": counts.most_common(10),
        "event_distribution": {
            "shift": {"frames": shift_frames, "pct": round(shift_frames/total_frames*100, 1)},
            "hold": {"frames": hold_frames, "pct": round(hold_frames/total_frames*100, 1)},
            "backchannel": {"frames": bc_frames, "pct": round(bc_frames/total_frames*100, 1)},
            "silence": {"frames": silence_frames, "pct": round(silence_frames/total_frames*100, 1)},
            "other": {"frames": other_frames, "pct": round(other_frames/total_frames*100, 1)},
        }
    }

    return stats
```

---

## 5. ASR Transcription (for Linguistic Branch)

### 5.1 Word-level Timestamps

```python
# scripts/04_transcribe.py

from faster_whisper import WhisperModel
import json

def transcribe_with_timestamps(audio_path, model_name="large-v3", device="cuda"):
    """
    Transcribe audio with word-level timestamps using faster-whisper.

    Returns:
        List[Dict] with keys: word, start, end
    """
    model = WhisperModel(model_name, device=device, compute_type="float16")

    segments, info = model.transcribe(
        audio_path,
        language="vi",
        word_timestamps=True,
        beam_size=5,
        vad_filter=True
    )

    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({
                "word": w.word.strip(),
                "start": round(w.start, 3),
                "end": round(w.end, 3)
            })

    return words
```

### 5.2 Aligning Text with Audio Frames

```python
# scripts/utils/text_alignment.py

import numpy as np
from typing import List, Dict

def align_text_to_frames(
    word_segments: List[Dict],
    total_frames: int,
    frame_rate: int = 50,
    asr_update_interval: float = 0.5
) -> Dict:
    """
    Create frame-aligned text features for the linguistic branch.

    The linguistic branch in MM-VAP-VI receives text updates
    every ~500ms (simulating streaming ASR). This function
    creates the alignment data.

    Args:
        word_segments: List of {word, start, end}
        total_frames: Total number of audio frames
        frame_rate: Frames per second
        asr_update_interval: How often ASR produces output (seconds)

    Returns:
        Dict with:
            - update_frames: List[int] — frame indices where text updates occur
            - cumulative_texts: List[str] — transcript up to each update point
            - frame_to_update_idx: np.ndarray (total_frames,) — maps each frame
              to its most recent text update index
    """
    total_duration = total_frames / frame_rate

    # Determine update points
    update_times = np.arange(0, total_duration, asr_update_interval)
    update_frames = (update_times * frame_rate).astype(int)

    # Build cumulative transcript at each update point
    cumulative_texts = []
    for t in update_times:
        # All words with end_time <= t
        text = " ".join(
            w["word"] for w in word_segments
            if w["end"] <= t
        )
        cumulative_texts.append(text)

    # Map each frame to its most recent update index
    frame_to_update_idx = np.zeros(total_frames, dtype=np.int32)
    for i, uf in enumerate(update_frames):
        if i + 1 < len(update_frames):
            frame_to_update_idx[uf:update_frames[i+1]] = i
        else:
            frame_to_update_idx[uf:] = i

    return {
        "update_frames": update_frames.tolist(),
        "cumulative_texts": cumulative_texts,
        "frame_to_update_idx": frame_to_update_idx
    }
```

---

## 6. Complete Pipeline Script

### 6.1 End-to-End Processing

```python
# scripts/build_vap_dataset.py (pseudocode / structure)

"""
Complete pipeline to build VAP dataset from raw audio files.

Usage:
    python scripts/build_vap_dataset.py \
        --input datasets/raw/youtube \
        --output datasets/vap \
        --hf-token <your_huggingface_token> \
        --device cuda
"""

import argparse
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# Import our modules
# from diarize import diarize_audio
# from voice_activity import build_voice_activity_matrix
# from generate_vap_labels import generate_vap_labels_vectorized
# from transcribe import transcribe_with_timestamps
# from text_alignment import align_text_to_frames


def process_single_file(
    audio_path: Path,
    output_dir: Path,
    diarization_pipeline,
    asr_model,
    num_speakers: int = 2,
    frame_rate: int = 50,
    sample_rate: int = 16000
):
    """
    Process a single audio file through the complete pipeline.

    Steps:
        1. Load & preprocess audio
        2. Speaker diarization
        3. Build voice activity matrix
        4. Generate VAP labels
        5. ASR transcription
        6. Text-frame alignment
        7. Save all outputs
    """
    file_id = audio_path.stem
    file_output_dir = output_dir / file_id
    file_output_dir.mkdir(parents=True, exist_ok=True)

    # === Step 1: Load audio ===
    audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    duration = len(audio) / sr
    total_frames = int(duration * frame_rate)

    print(f"  Audio: {duration:.1f}s, {total_frames} frames")

    # Save preprocessed audio
    sf.write(str(file_output_dir / "audio.wav"), audio, sr)

    # === Step 2: Speaker diarization ===
    diarization_segments = diarize_audio(
        str(audio_path),
        diarization_pipeline,
        num_speakers=num_speakers
    )

    print(f"  Diarization: {len(diarization_segments)} segments")

    # === Step 3: Voice activity matrix ===
    va_matrix, speaker_map = build_voice_activity_matrix(
        diarization_segments,
        duration,
        frame_rate=frame_rate,
        num_speakers=num_speakers
    )

    np.save(str(file_output_dir / "voice_activity.npy"), va_matrix)
    print(f"  VA matrix: {va_matrix.shape}")

    # === Step 4: VAP labels ===
    vap_labels = generate_vap_labels_vectorized(
        va_matrix,
        frame_rate=frame_rate
    )

    np.save(str(file_output_dir / "vap_labels.npy"), vap_labels)
    print(f"  VAP labels: {vap_labels.shape}, "
          f"{len(set(vap_labels.tolist()))} unique classes")

    # === Step 5: ASR transcription ===
    word_segments = transcribe_with_timestamps(
        str(audio_path),
        asr_model
    )

    print(f"  ASR: {len(word_segments)} words")

    # === Step 6: Text-frame alignment ===
    text_alignment = align_text_to_frames(
        word_segments,
        total_frames,
        frame_rate=frame_rate
    )

    # === Step 7: Save metadata ===
    metadata = {
        "file_id": file_id,
        "audio_file": audio_path.name,
        "duration_sec": round(duration, 2),
        "sample_rate": sr,
        "frame_rate": frame_rate,
        "total_frames": total_frames,
        "num_speakers": num_speakers,
        "speaker_map": speaker_map,
        "diarization_segments": diarization_segments,
        "num_words": len(word_segments),
        "num_text_updates": len(text_alignment["update_frames"]),
    }

    with open(file_output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Save transcript
    transcript_data = {
        "words": word_segments,
        "alignment": {
            "update_frames": text_alignment["update_frames"],
            "cumulative_texts": text_alignment["cumulative_texts"],
        }
    }
    with open(file_output_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)

    # Save frame_to_update_idx separately (large array)
    np.save(
        str(file_output_dir / "frame_to_update_idx.npy"),
        text_alignment["frame_to_update_idx"]
    )

    return metadata


def build_dataset(args):
    """Main pipeline: process all audio files."""

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find audio files
    audio_files = sorted(
        list(input_dir.glob("*.wav")) +
        list(input_dir.glob("*.mp3")) +
        list(input_dir.glob("*.m4a"))
    )

    print(f"Found {len(audio_files)} audio files")

    # Load models once
    # diarization_pipeline = load_diarization_pipeline(args.hf_token)
    # asr_model = load_asr_model(args.device)

    all_metadata = []

    for i, audio_file in enumerate(audio_files):
        print(f"\n[{i+1}/{len(audio_files)}] Processing: {audio_file.name}")

        try:
            metadata = process_single_file(
                audio_file,
                output_dir,
                diarization_pipeline=None,  # placeholder
                asr_model=None,             # placeholder
                num_speakers=args.num_speakers,
            )
            all_metadata.append(metadata)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save dataset manifest
    manifest = {
        "total_files": len(all_metadata),
        "total_duration_hours": round(
            sum(m["duration_sec"] for m in all_metadata) / 3600, 1
        ),
        "total_frames": sum(m["total_frames"] for m in all_metadata),
        "files": [m["file_id"] for m in all_metadata],
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nDataset built: {manifest['total_files']} files, "
          f"{manifest['total_duration_hours']}h")
```

### 6.2 Output Directory Structure

```
datasets/vap/
├── manifest.json                    # Dataset metadata
├── KH_TUY_T_V_I/
│   ├── audio.wav                    # 16kHz mono audio
│   ├── voice_activity.npy           # (2, T) float32
│   ├── vap_labels.npy               # (T,) int64, values [0-255]
│   ├── transcript.json              # word-level timestamps + alignment
│   ├── frame_to_update_idx.npy      # (T,) int32
│   └── metadata.json                # file metadata
├── PODCAST_ABC/
│   ├── ...
│   └── ...
└── ...
```

---

## 7. Train/Val/Test Split

### 7.1 Split Strategy

```python
# scripts/split_vap_dataset.py

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

def split_dataset(
    dataset_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split VAP dataset by conversation file.

    Rules:
        - All frames from one conversation go to same split
        - No speaker leakage between splits (ideal, best-effort)
        - Stratify by conversation type if metadata available
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    with open(dataset_dir / "manifest.json") as f:
        manifest = json.load(f)

    file_ids = manifest["files"]
    random.seed(seed)
    random.shuffle(file_ids)

    n = len(file_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": file_ids[:n_train],
        "val": file_ids[n_train:n_train + n_val],
        "test": file_ids[n_train + n_val:]
    }

    # Save split manifest
    split_manifest = {"seed": seed, "splits": {}}
    for split_name, ids in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        total_frames = 0
        total_duration = 0

        for fid in ids:
            # Symlink or copy data
            src = dataset_dir / fid
            dst = split_dir / fid
            if not dst.exists():
                # Use symlink on Linux, copy on Windows
                try:
                    dst.symlink_to(src)
                except OSError:
                    shutil.copytree(str(src), str(dst))

            # Read metadata
            with open(src / "metadata.json") as f:
                meta = json.load(f)
            total_frames += meta["total_frames"]
            total_duration += meta["duration_sec"]

        split_manifest["splits"][split_name] = {
            "files": ids,
            "num_files": len(ids),
            "total_frames": total_frames,
            "total_duration_hours": round(total_duration / 3600, 2)
        }

    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)

    # Print summary
    for name, info in split_manifest["splits"].items():
        print(f"  {name}: {info['num_files']} files, "
              f"{info['total_duration_hours']}h, "
              f"{info['total_frames']} frames")
```

---

## 8. Dataset Class (PyTorch)

### 8.1 VAPDataset

```python
# src/data/vap_dataset.py

import torch
import numpy as np
import json
import librosa
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple


class VAPDataset(Dataset):
    """
    VAP Dataset for MM-VAP-VI.

    Produces overlapping windows of audio + VAP labels + text alignment.
    Each sample is a fixed-length window from a conversation.

    Directory structure expected:
        split_dir/
        ├── conversation_1/
        │   ├── audio.wav
        │   ├── vap_labels.npy
        │   ├── transcript.json
        │   ├── frame_to_update_idx.npy
        │   └── metadata.json
        └── conversation_2/
            └── ...
    """

    def __init__(
        self,
        split_dir: str,
        window_sec: float = 20.0,
        stride_sec: float = 5.0,
        frame_rate: int = 50,
        sample_rate: int = 16000,
        include_text: bool = True,
    ):
        self.split_dir = Path(split_dir)
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.include_text = include_text

        self.window_frames = int(window_sec * frame_rate)
        self.stride_frames = int(stride_sec * frame_rate)
        self.window_samples = int(window_sec * sample_rate)

        # Build index: list of (conversation_dir, start_frame)
        self.samples = []
        self._build_index()

    def _build_index(self):
        """Build list of (conversation_dir, start_frame) tuples."""
        conv_dirs = sorted([
            d for d in self.split_dir.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ])

        for conv_dir in conv_dirs:
            with open(conv_dir / "metadata.json") as f:
                meta = json.load(f)

            total_frames = meta["total_frames"]

            # Generate overlapping windows
            start = 0
            while start + self.window_frames <= total_frames:
                self.samples.append((conv_dir, start))
                start += self.stride_frames

            # Include last partial window if it's at least half the window size
            if start < total_frames and (total_frames - start) > self.window_frames // 2:
                self.samples.append((conv_dir, total_frames - self.window_frames))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        conv_dir, start_frame = self.samples[idx]

        end_frame = start_frame + self.window_frames
        start_sec = start_frame / self.frame_rate
        duration_sec = self.window_sec

        # === Load audio ===
        audio, _ = librosa.load(
            str(conv_dir / "audio.wav"),
            sr=self.sample_rate,
            offset=start_sec,
            duration=duration_sec
        )

        # Pad if needed
        if len(audio) < self.window_samples:
            audio = np.pad(audio, (0, self.window_samples - len(audio)))

        audio = audio[:self.window_samples]  # truncate if needed

        # === Load VAP labels ===
        vap_labels = np.load(conv_dir / "vap_labels.npy")
        window_labels = vap_labels[start_frame:end_frame]

        # Pad if needed
        if len(window_labels) < self.window_frames:
            window_labels = np.pad(
                window_labels,
                (0, self.window_frames - len(window_labels)),
                constant_values=0  # silence class
            )

        # === Create padding mask ===
        # True for valid frames, False for padding
        valid_frames = min(end_frame, len(vap_labels)) - start_frame
        padding_mask = np.ones(self.window_frames, dtype=bool)
        padding_mask[valid_frames:] = False

        result = {
            "audio": torch.from_numpy(audio).float(),
            "vap_labels": torch.from_numpy(window_labels).long(),
            "padding_mask": torch.from_numpy(padding_mask).bool(),
        }

        # === Load text alignment (optional) ===
        if self.include_text:
            frame_to_update = np.load(conv_dir / "frame_to_update_idx.npy")
            window_update_idx = frame_to_update[start_frame:end_frame]

            with open(conv_dir / "transcript.json") as f:
                transcript = json.load(f)

            cumulative_texts = transcript["alignment"]["cumulative_texts"]

            # Get unique update indices in this window
            unique_updates = sorted(set(window_update_idx.tolist()))

            # Get texts for each update
            texts_in_window = []
            for ui in unique_updates:
                if ui < len(cumulative_texts):
                    texts_in_window.append(cumulative_texts[ui])
                else:
                    texts_in_window.append("")

            # Map window frames to local update index
            local_update_map = {ui: i for i, ui in enumerate(unique_updates)}
            local_frame_to_update = np.array([
                local_update_map[ui] for ui in window_update_idx
            ], dtype=np.int32)

            # Pad if needed
            if len(local_frame_to_update) < self.window_frames:
                last_val = local_frame_to_update[-1] if len(local_frame_to_update) > 0 else 0
                local_frame_to_update = np.pad(
                    local_frame_to_update,
                    (0, self.window_frames - len(local_frame_to_update)),
                    constant_values=last_val
                )

            result["texts"] = texts_in_window
            result["frame_to_text_idx"] = torch.from_numpy(local_frame_to_update).long()

        # === Metadata ===
        result["metadata"] = {
            "conv_dir": str(conv_dir.name),
            "start_frame": start_frame,
            "start_sec": round(start_sec, 2),
        }

        return result

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate function for DataLoader."""
        result = {
            "audio": torch.stack([b["audio"] for b in batch]),
            "vap_labels": torch.stack([b["vap_labels"] for b in batch]),
            "padding_mask": torch.stack([b["padding_mask"] for b in batch]),
        }

        if "texts" in batch[0]:
            # Texts are variable-length lists — keep as list of lists
            result["texts"] = [b["texts"] for b in batch]
            result["frame_to_text_idx"] = torch.stack(
                [b["frame_to_text_idx"] for b in batch]
            )

        result["metadata"] = [b["metadata"] for b in batch]

        return result


def create_vap_dataloader(
    split_dir: str,
    window_sec: float = 20.0,
    stride_sec: float = 5.0,
    batch_size: int = 16,
    num_workers: int = 4,
    include_text: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for the VAP dataset."""
    dataset = VAPDataset(
        split_dir=split_dir,
        window_sec=window_sec,
        stride_sec=stride_sec,
        include_text=include_text,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=VAPDataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )
```

---

## 9. Data Validation & Quality Checks

### 9.1 Diarization Quality Check

```python
def check_diarization_quality(va_matrix, metadata):
    """
    Verify diarization output makes sense.

    Red flags:
        - One speaker has < 10% of total speech → probably wrong num_speakers
        - > 30% overlap → diarization might be confused
        - > 50% silence → audio might have long non-speech segments
    """
    sp1_active = va_matrix[0].mean()
    sp2_active = va_matrix[1].mean()
    overlap = ((va_matrix[0] > 0) & (va_matrix[1] > 0)).mean()
    silence = ((va_matrix[0] == 0) & (va_matrix[1] == 0)).mean()

    issues = []
    if min(sp1_active, sp2_active) < 0.1:
        issues.append(f"Speaker imbalance: SP1={sp1_active:.1%}, SP2={sp2_active:.1%}")
    if overlap > 0.3:
        issues.append(f"High overlap: {overlap:.1%}")
    if silence > 0.5:
        issues.append(f"High silence: {silence:.1%}")

    return {
        "sp1_activity": round(sp1_active, 3),
        "sp2_activity": round(sp2_active, 3),
        "overlap": round(overlap, 3),
        "silence": round(silence, 3),
        "issues": issues,
        "quality": "good" if not issues else "needs_review"
    }
```

### 9.2 VAP Label Sanity Check

```python
def check_vap_labels(labels, va_matrix, frame_rate=50):
    """
    Verify VAP labels are consistent with voice activity matrix.

    Checks:
        - Label at silence region should be class 0 (if far from speech)
        - Hold labels should appear during sustained speech
        - Shift labels should appear near turn boundaries
    """
    issues = []

    # Check class 0 (all silent) frequency
    silence_label_pct = (labels == 0).mean()
    actual_silence_pct = ((va_matrix[0] == 0) & (va_matrix[1] == 0)).mean()

    # These should be roughly correlated
    # (not exact because labels look 2s ahead)
    if abs(silence_label_pct - actual_silence_pct) > 0.3:
        issues.append(
            f"Silence mismatch: label_silence={silence_label_pct:.1%}, "
            f"actual_silence={actual_silence_pct:.1%}"
        )

    # Check that not all labels are the same
    unique_labels = len(set(labels.tolist()))
    if unique_labels < 5:
        issues.append(f"Too few unique labels: {unique_labels}")

    return {
        "unique_labels": unique_labels,
        "silence_label_pct": round(silence_label_pct, 3),
        "actual_silence_pct": round(actual_silence_pct, 3),
        "issues": issues,
        "quality": "good" if not issues else "needs_review"
    }
```

---

## 10. Tóm tắt Pipeline hoàn chỉnh

```
=== PIPELINE MỚI CHO MM-VAP-VI ===

Input: Raw audio files (YouTube podcasts, 16kHz mono)

Step 1: Download & preprocess audio
        yt-dlp → librosa resample → 16kHz mono WAV

Step 2: Speaker diarization
        pyannote/speaker-diarization-3.1 → segments per speaker

Step 3: Voice activity matrix
        Diarization segments → binary matrix (2, T) at 50fps

Step 4: VAP label generation
        VA matrix → look-ahead 2s → 4 bins × 2 speakers → 256-class per frame

Step 5: ASR transcription
        faster-whisper → word-level timestamps

Step 6: Text-frame alignment
        Words + timestamps → cumulative text every 500ms → frame mapping

Step 7: Validate & save
        Quality checks → save .npy, .json, .wav per conversation

Step 8: Train/val/test split
        Split by conversation file (80/10/10)

Output:
  datasets/vap/
  ├── train/
  │   ├── conv_001/ {audio.wav, vap_labels.npy, voice_activity.npy, ...}
  │   └── conv_002/ ...
  ├── val/
  │   └── ...
  ├── test/
  │   └── ...
  └── split_manifest.json
```
