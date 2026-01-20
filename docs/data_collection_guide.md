# 🎯 Hướng dẫn Xây dựng Dataset Turn-Taking cho Tiếng Việt

## Tổng quan Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CUSTOM DATASET PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. THU THẬP        2. XỬ LÝ           3. GÁN NHÃN        4. CHUẨN HÓA      │
│  ───────────        ────────           ──────────         ─────────         │
│                                                                              │
│  YouTube/Podcast → Diarization →  LLM Labeling →  Train/Val/Test           │
│  (50-100 hours)    + ASR            (Gemini)         Split                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Thu thập dữ liệu (Data Collection)

### Nguồn dữ liệu đề xuất

| Nguồn | Loại | Ưu điểm | Link |
|-------|------|---------|------|
| **YouTube Interviews** | Video/Audio | Dễ crawl, nhiều chủ đề | yt-dlp |
| **Vietcetera Podcast** | Audio | Chất lượng cao, 2 người | Spotify/RSS |
| **Radio VOV** | Audio | Hội thoại tự nhiên | Website |

### Tiêu chí chọn video/audio:
- ✅ **2 người** nói chuyện (tốt nhất)
- ✅ Âm thanh rõ ràng, ít nhiễu
- ✅ Hội thoại tự nhiên (không đọc kịch bản)
- ✅ Độ dài 10-60 phút mỗi episode
- ❌ Tránh: Đọc tin tức, thuyết trình 1 người

### Script crawl YouTube:

```bash
# Cài đặt
pip install yt-dlp

# Download audio từ playlist/channel
yt-dlp --extract-audio --audio-format wav --audio-quality 0 \
    -o "data/raw/youtube/%(title)s.%(ext)s" \
    "https://www.youtube.com/playlist?list=PLxxxxxx"

# Hoặc từ video đơn lẻ
yt-dlp -x --audio-format wav "https://www.youtube.com/watch?v=xxxxx"
```

### Mục tiêu: 50-100 giờ audio hội thoại

---

## Phase 2: Xử lý Audio (Processing)

### 2.1 Speaker Diarization (Tách người nói)

**Tool:** `pyannote-audio` - SOTA speaker diarization

```python
# Cài đặt
pip install pyannote.audio

# Code diarization
from pyannote.audio import Pipeline

# Cần Hugging Face token (miễn phí)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN"
)

# Chạy diarization
diarization = pipeline("audio.wav")

# Output: ai nói lúc nào
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}s - {turn.end:.1f}s: {speaker}")
    # 0.0s - 2.5s: SPEAKER_00
    # 2.7s - 5.1s: SPEAKER_01
    # 5.3s - 8.2s: SPEAKER_00
```

### 2.2 ASR Transcription (Chuyển giọng nói thành text)

**Tool:** `PhoWhisper-base` - SOTA Vietnamese ASR

```python
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-base",
    chunk_length_s=30,
    return_timestamps=True  # Quan trọng!
)

result = asr("audio.wav")

# Output với timestamps
for chunk in result["chunks"]:
    print(f"{chunk['timestamp'][0]:.1f}s: {chunk['text']}")
    # 0.0s: "Anh đi đâu đấy nhỉ"
    # 2.8s: "Ừ anh đi chợ mua đồ"
```

### 2.3 Merge Diarization + ASR

```python
def merge_diarization_asr(diarization, asr_result):
    """Kết hợp ai nói + nói gì"""
    segments = []
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Tìm text tương ứng với khoảng thời gian này
        text = ""
        for chunk in asr_result["chunks"]:
            chunk_start = chunk["timestamp"][0]
            chunk_end = chunk["timestamp"][1] or chunk_start + 1
            
            # Nếu chunk nằm trong turn này
            if chunk_start >= turn.start and chunk_end <= turn.end:
                text += chunk["text"] + " "
        
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
            "text": text.strip()
        })
    
    return segments
```

**Output mẫu:**
```json
[
    {"speaker": "A", "start": 0.0, "end": 2.5, "text": "Anh đi đâu đấy nhỉ"},
    {"speaker": "B", "start": 2.7, "end": 5.1, "text": "Ừ anh đi chợ mua đồ"},
    {"speaker": "A", "start": 5.3, "end": 8.2, "text": "Vậy mua giúp em ít rau nhé"}
]
```

---

## Phase 3: Gán nhãn Turn-Taking (LLM Labeling)

### Tại sao dùng LLM?

```
TRƯỚC: Gán nhãn thủ công → Tốn 100+ giờ cho 50h audio
SAU:   LLM-as-Judge      → Tự động, chỉ cần review 10%
```

### Prompt cho Gemini:

```python
import google.generativeai as genai

PROMPT = """Bạn là chuyên gia ngôn ngữ học hội thoại tiếng Việt.

Phân tích đoạn hội thoại sau và gán nhãn cho MỖI PHÁT NGÔN:

- YIELD: Người nói KẾT THÚC, sẵn sàng nhường lời
  (Dấu hiệu: hư từ cuối câu như "nhé", "nhỉ", "à", "hả", "ạ", giọng đi xuống)

- HOLD: Người nói CHƯA XONG, sẽ tiếp tục
  (Dấu hiệu: câu còn treo, có "mà", "thì", "là", "vì", giọng treo)

- BACKCHANNEL: Phản hồi ngắn KHÔNG chiếm lượt
  (Ví dụ: "ừ", "vâng", "ờ", "à", "thế à", "vậy hả")

HỘI THOẠI:
{conversation}

Trả về JSON:
[
  {{"speaker": "A", "text": "...", "label": "YIELD/HOLD/BACKCHANNEL", "reason": "..."}}
]
"""

def label_conversation(segments):
    conversation = "\n".join([
        f"[{s['speaker']}] ({s['start']:.1f}s): {s['text']}"
        for s in segments
    ])
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        PROMPT.format(conversation=conversation),
        generation_config={"response_mime_type": "application/json"}
    )
    
    return json.loads(response.text)
```

### Quality Control (Kiểm tra chất lượng):

```python
# Sau khi LLM gán nhãn, kiểm tra tự động
def validate_labels(labeled_segments):
    issues = []
    
    for seg in labeled_segments:
        text = seg["text"].lower()
        label = seg["label"]
        
        # Rule-based validation
        if label == "YIELD" and any(h in text for h in ["mà", "thì", "vì"]):
            issues.append(f"Possible HOLD mislabeled as YIELD: {text}")
        
        if label == "BACKCHANNEL" and len(text.split()) > 5:
            issues.append(f"Long text labeled as BACKCHANNEL: {text}")
    
    return issues
```

---

## Phase 4: Chuẩn bị Dataset cuối cùng

### Cấu trúc thư mục:

```
data/
├── raw/                          # Audio gốc
│   └── youtube/
│       ├── interview_001.wav
│       └── interview_002.wav
├── processed/
│   ├── diarization/              # Speaker segments
│   │   └── interview_001.json
│   ├── transcripts/              # ASR output
│   │   └── interview_001.json
│   └── labeled/                  # Final labels
│       └── interview_001.json
└── final/
    ├── train.json                # 80%
    ├── val.json                  # 10%
    └── test.json                 # 10%
```

### Format dữ liệu cuối:

```json
{
  "audio_file": "interview_001.wav",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "speaker": "A",
      "text": "Anh đi đâu đấy nhỉ",
      "label": "YIELD",
      "audio_features": "processed/features/interview_001_seg_0.pt"
    }
  ]
}
```

---

## Tóm tắt Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                    COMPLETE WORKFLOW                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. yt-dlp         Download YouTube/Podcast audio                │
│       ↓                                                           │
│  2. pyannote       Speaker diarization (ai nói lúc nào)          │
│       ↓                                                           │
│  3. PhoWhisper     ASR transcription (nói gì)                    │
│       ↓                                                           │
│  4. Merge          Kết hợp speaker + text + timestamp            │
│       ↓                                                           │
│  5. Gemini         LLM gán nhãn YIELD/HOLD/BACKCHANNEL           │
│       ↓                                                           │
│  6. Validate       Rule-based QC + human review 10%              │
│       ↓                                                           │
│  7. Split          Train/Val/Test                                 │
│       ↓                                                           │
│  8. Features       Trích xuất Mel + F0 + Energy                  │
│                                                                   │
│  OUTPUT: Dataset sẵn sàng cho training                           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Thời gian ước tính:
| Bước | Thời gian (50h audio) |
|------|----------------------|
| Crawl | 2-3 giờ |
| Diarization | ~5 giờ (GPU) |
| ASR | ~3 giờ (GPU) |
| LLM Labeling | ~2 giờ (API) |
| **Tổng** | **~12 giờ** |

---

## Yêu cầu phần cứng/API:

| Resource | Requirement |
|----------|-------------|
| GPU | Recommended (RTX 3060+) |
| HuggingFace Token | Free (cho pyannote) |
| Google API Key | Free tier đủ dùng |
| Storage | ~100GB cho 50h audio |
