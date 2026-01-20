# 🔧 Semi-Auto + Review Pipeline cho Turn-Taking Dataset

## Tổng quan Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEMI-AUTO + REVIEW PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   PHASE A: AUTO              PHASE B: REVIEW           PHASE C: FINAL       │
│   ═════════════              ═════════════            ═════════════          │
│                                                                              │
│   YouTube/Podcast     →     Label Studio      →     Quality Dataset         │
│        ↓                         ↓                        ↓                 │
│   whisperX/pyannote        Human Review            Train/Val/Test           │
│        ↓                         ↓                                          │
│   LLM Pre-labeling          Corrections                                     │
│                                                                              │
│   Time: ~2h/50h audio      Time: ~10h/50h audio    Time: ~1h               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase A: Auto-Processing

### A1. Thu thập Audio (2-3 giờ)

```bash
# Download từ YouTube (ví dụ podcast interview)
pip install yt-dlp

# Download playlist
yt-dlp --extract-audio --audio-format wav -o "data/raw/%(title)s.%(ext)s" \
    "https://www.youtube.com/playlist?list=YOUR_PLAYLIST"
```

### A2. Auto Diarization + ASR (5 giờ GPU cho 50h audio)

**Dùng whisperX** (tích hợp Whisper + Diarization):

```python
# scripts/auto_process.py
import whisperx
import json
from pathlib import Path

def process_audio(audio_path: str, output_dir: str):
    """Auto diarization + ASR với whisperX"""
    
    # Load model
    device = "cuda"  # hoặc "cpu"
    model = whisperx.load_model("large-v3", device)
    
    # Load audio
    audio = whisperx.load_audio(audio_path)
    
    # 1. ASR
    result = model.transcribe(audio, language="vi")
    
    # 2. Align timestamps
    model_a, metadata = whisperx.load_align_model(
        language_code="vi", device=device
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device
    )
    
    # 3. Diarization
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token="YOUR_HF_TOKEN"
    )
    diarize_segments = diarize_model(audio)
    
    # 4. Assign speakers
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    # Save
    output_path = Path(output_dir) / f"{Path(audio_path).stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result

# Chạy cho tất cả files
for audio_file in Path("data/raw").glob("*.wav"):
    process_audio(str(audio_file), "data/processed/auto")
```

### A3. LLM Pre-labeling (2 giờ API)

```python
# scripts/llm_prelabel.py
import json
import google.generativeai as genai

PROMPT = """Phân tích hội thoại và gán nhãn turn-taking:
- YIELD: Kết thúc lượt nói (hư từ: nhé, nhỉ, ạ, hả)
- HOLD: Chưa xong (hư từ: mà, thì, là, vì)  
- BACKCHANNEL: Phản hồi ngắn (ừ, vâng, ờ)

{conversation}

Output JSON: [{"segment_id": 0, "label": "YIELD/HOLD/BACKCHANNEL", "confidence": 0.9}]"""

def prelabel_conversation(segments):
    conversation = "\n".join([
        f"[{s.get('speaker', '?')}] {s['text']}" 
        for s in segments
    ])
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        PROMPT.format(conversation=conversation),
        generation_config={"response_mime_type": "application/json"}
    )
    
    return json.loads(response.text)
```

**Output sau Phase A:**
```json
{
  "audio_file": "interview_001.wav",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "speaker": "SPEAKER_00",
      "text": "Anh đi đâu đấy nhỉ",
      "auto_label": "YIELD",
      "confidence": 0.85,
      "needs_review": false
    },
    {
      "id": 1,
      "start": 2.7,
      "end": 2.9,
      "speaker": "SPEAKER_01",
      "text": "Ừ",
      "auto_label": "BACKCHANNEL",
      "confidence": 0.6,
      "needs_review": true  // Low confidence → flag for review
    }
  ]
}
```

---

## Phase B: Human Review với Label Studio

### B1. Setup Label Studio

```bash
# Cài đặt
pip install label-studio

# Khởi động
label-studio start --port 8080
```

Truy cập: http://localhost:8080

### B2. Tạo Project

1. **Create Project** → "Viet-Turn Review"
2. **Labeling Setup** → Custom Template:

```xml
<View>
  <Header value="Audio Segment"/>
  <Audio name="audio" value="$audio_url"/>
  
  <Header value="Transcript"/>
  <Text name="transcript" value="$text"/>
  
  <Header value="Speaker"/>
  <Text name="speaker" value="$speaker"/>
  
  <Header value="Auto Label (confidence: $confidence)"/>
  <Text name="auto_label" value="$auto_label"/>
  
  <Header value="Your Label"/>
  <Choices name="turn_label" toName="audio" choice="single">
    <Choice value="YIELD" hint="Kết thúc lượt - nhé, nhỉ, ạ"/>
    <Choice value="HOLD" hint="Chưa xong - mà, thì, là"/>
    <Choice value="BACKCHANNEL" hint="Phản hồi ngắn - ừ, vâng"/>
  </Choices>
  
  <Header value="Issues (optional)"/>
  <Choices name="issues" toName="audio" choice="multiple">
    <Choice value="WRONG_SPEAKER"/>
    <Choice value="WRONG_TEXT"/>
    <Choice value="OVERLAP"/>
    <Choice value="NOISE"/>
  </Choices>
</View>
```

### B3. Import Data

```python
# scripts/export_to_labelstudio.py
import json

def export_for_labelstudio(processed_dir, output_file):
    tasks = []
    
    for json_file in Path(processed_dir).glob("*.json"):
        data = json.load(open(json_file))
        
        for seg in data["segments"]:
            # Chỉ review segments có confidence thấp hoặc cần check
            if seg.get("needs_review", False) or seg.get("confidence", 1) < 0.7:
                tasks.append({
                    "data": {
                        "audio_url": f"/data/audio/{data['audio_file']}",
                        "text": seg["text"],
                        "speaker": seg["speaker"],
                        "auto_label": seg["auto_label"],
                        "confidence": seg["confidence"],
                        "segment_id": seg["id"],
                        "source_file": json_file.name
                    }
                })
    
    with open(output_file, "w") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    
    print(f"Exported {len(tasks)} tasks for review")

export_for_labelstudio("data/processed/auto", "data/labelstudio_tasks.json")
```

### B4. Review Guidelines

```
┌─────────────────────────────────────────────────────────────────┐
│                    REVIEW CHECKLIST                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. NGHE AUDIO trước khi đọc text                               │
│                                                                  │
│  2. KIỂM TRA:                                                    │
│     □ Speaker đúng chưa?                                        │
│     □ Text đúng chưa? (đặc biệt hư từ cuối câu)                │
│     □ Có overlap không?                                         │
│                                                                  │
│  3. GÁN NHÃN:                                                    │
│     • YIELD: Giọng đi xuống + hư từ kết thúc                   │
│     • HOLD: Giọng treo + câu chưa xong                         │
│     • BACKCHANNEL: Ngắn + không chiếm lượt                     │
│                                                                  │
│  4. FLAG ISSUES nếu có vấn đề                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### B5. Export & Merge

```python
# scripts/merge_reviewed.py
def merge_labels(auto_dir, reviewed_export, output_dir):
    """Merge auto labels với reviewed corrections"""
    
    # Load reviewed
    reviewed = json.load(open(reviewed_export))
    reviewed_map = {
        (r["source_file"], r["segment_id"]): r["label"]
        for r in reviewed
    }
    
    # Merge
    for json_file in Path(auto_dir).glob("*.json"):
        data = json.load(open(json_file))
        
        for seg in data["segments"]:
            key = (json_file.name, seg["id"])
            if key in reviewed_map:
                seg["label"] = reviewed_map[key]  # Use reviewed
                seg["reviewed"] = True
            else:
                seg["label"] = seg["auto_label"]  # Keep auto
                seg["reviewed"] = False
        
        # Save
        output_path = Path(output_dir) / json_file.name
        json.dump(data, open(output_path, "w"), ensure_ascii=False, indent=2)
```

---

## Phase C: Final Dataset

### C1. Quality Check

```python
# scripts/quality_check.py
def check_quality(data_dir):
    stats = {"total": 0, "reviewed": 0, "by_label": {}}
    issues = []
    
    for json_file in Path(data_dir).glob("*.json"):
        data = json.load(open(json_file))
        
        for seg in data["segments"]:
            stats["total"] += 1
            if seg.get("reviewed"):
                stats["reviewed"] += 1
            
            label = seg["label"]
            stats["by_label"][label] = stats["by_label"].get(label, 0) + 1
            
            # Check issues
            if label == "BACKCHANNEL" and len(seg["text"].split()) > 5:
                issues.append(f"Long BACKCHANNEL: {seg['text']}")
    
    return stats, issues
```

### C2. Train/Val/Test Split

```python
# scripts/split_dataset.py
from sklearn.model_selection import train_test_split

def split_dataset(data_dir, output_dir, test_size=0.1, val_size=0.1):
    all_files = list(Path(data_dir).glob("*.json"))
    
    # Split by file (không split trong file)
    train_files, test_files = train_test_split(all_files, test_size=test_size)
    train_files, val_files = train_test_split(train_files, test_size=val_size/(1-test_size))
    
    # Save splits
    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        split_data = []
        for f in files:
            split_data.extend(json.load(open(f))["segments"])
        
        with open(Path(output_dir) / f"{split_name}.json", "w") as out:
            json.dump(split_data, out, ensure_ascii=False, indent=2)
        
        print(f"{split_name}: {len(split_data)} segments from {len(files)} files")
```

---

## Timeline ước tính

| Phase | Task | Time (50h audio) |
|-------|------|------------------|
| A | Download audio | 2-3h |
| A | whisperX processing | 5h (GPU) |
| A | LLM pre-labeling | 2h |
| B | Setup Label Studio | 1h |
| B | Human review (~20% data) | **8-10h** |
| C | Merge + QC | 1h |
| **Total** | | **~20h** |

---

## Thư mục Project

```
data/
├── raw/                    # Audio gốc
├── processed/
│   ├── auto/              # Output từ whisperX + LLM
│   └── reviewed/          # Sau khi merge review
├── labelstudio/           # Export từ Label Studio
└── final/
    ├── train.json
    ├── val.json
    └── test.json
```
