# 📋 Hướng Dẫn Collect Data - Viet-TurnEdge

## 📦 Bước 0: Cài Đặt Dependencies

```bash
cd f:\Viet-Turn

# Cài đặt requirements
pip install -r requirements.txt

# Cài thêm whisperx (cần cho script 02)
pip install whisperx

# Cài yt-dlp (cần cho script 01)
pip install yt-dlp
```

## 🔑 Bước 1: Setup API Keys

1. **Tạo file `.env`** từ template:
```bash
copy .env.example .env
```

2. **Điền API keys vào `.env`**:
```env
# HuggingFace token (cho speaker diarization)
# Lấy tại: https://huggingface.co/settings/tokens
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxx

# Google API key (cho Gemini labeling)
# Lấy tại: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 📥 Bước 2: Download Audio (Script 01)

### Option A: Download từ YouTube
```bash
# Download 1 video
python scripts/01_download_audio.py --url "https://youtube.com/watch?v=xxxxx" --output data/raw/youtube

# Download playlist
python scripts/01_download_audio.py --playlist "https://youtube.com/playlist?list=xxxxx" --output data/raw/youtube --max-videos 10
```

### Option B: Download từ danh sách URLs
```bash
# Tạo file urls.txt
echo "https://youtube.com/watch?v=video1" > urls.txt
echo "https://youtube.com/watch?v=video2" >> urls.txt

# Download
python scripts/01_download_audio.py --file urls.txt --output data/raw/youtube
```

### 💡 Tips nguồn audio tốt:
- Podcast tiếng Việt (nhiều đối thoại)
- Phỏng vấn (2 người nói)
- Talk show (nhiều người nói)

---

## 🎤 Bước 3: ASR + Diarization (Script 02)

```bash
# Process tất cả audio trong thư mục
python scripts/02_auto_process.py --input data/raw/youtube --output data/processed/auto

# Chỉ chạy ASR (không cần HF token)
python scripts/02_auto_process.py --input data/raw/youtube --output data/processed/auto --skip-diarization

# Dùng model nhỏ hơn (nhanh hơn)
python scripts/02_auto_process.py --input data/raw/youtube --output data/processed/auto --model small
```

### Output format (`data/processed/auto/video_name.json`):
```json
{
  "audio_file": "video_name.wav",
  "duration": 3600.5,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Xin chào các bạn",
      "speaker": "SPEAKER_00"
    }
  ]
}
```

---

## 🤖 Bước 4: LLM Pre-labeling (Script 03)

```bash
# Label với Gemini API
python scripts/03_llm_prelabel.py --input data/processed/auto --output data/processed/labeled

# Chỉ dùng rule-based (không cần API)
python scripts/03_llm_prelabel.py --input data/processed/auto --output data/processed/labeled --no-llm
```

### Output thêm các fields:
```json
{
  "auto_label": "YIELD",      // YIELD / HOLD / BACKCHANNEL
  "confidence": 0.85,
  "label_reason": "có 'nhỉ' cuối câu",
  "needs_review": false
}
```

---

## 📋 Bước 5: Export cho Label Studio (Script 04)

```bash
python scripts/04_export_labelstudio.py --input data/processed/labeled --output data/labelstudio

# Export tất cả (không chỉ cần review)
python scripts/04_export_labelstudio.py --input data/processed/labeled --output data/labelstudio --all
```

### Setup Label Studio:
```bash
# Cài Label Studio
pip install label-studio

# Start server
label-studio start --port 8080

# Mở browser: http://localhost:8080
```

### Import vào Label Studio:
1. Create Project → Import → Upload `data/labelstudio/tasks.json`
2. Settings → Labeling Interface → Code → Paste nội dung từ `data/labelstudio/labeling_config.xml`
3. Start labeling!

---

## ✅ Bước 6: Merge Reviewed Labels (Script 05)

Sau khi review xong trên Label Studio:
1. Export annotations: Export → JSON
2. Save file về: `data/labelstudio/export.json`
3. Chạy merge:

```bash
python scripts/05_merge_reviewed.py \
  --auto data/processed/labeled \
  --reviewed data/labelstudio/export.json \
  --output data/processed/final \
  --validate
```

---

## 📊 Bước 7: Split Dataset (Script 06)

```bash
python scripts/06_split_dataset.py \
  --input data/processed/final \
  --output data/final \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

### Output:
```
data/final/
├── train.json      # 80%
├── val.json        # 10%
├── test.json       # 10%
└── manifest.json   # Metadata
```

---

## ✂️ Bước 8: Cut Audio Segments (Script 07)

```bash
python scripts/07_cut_segments.py \
  --input data/final \
  --audio-dir data/raw/youtube \
  --output data/segments
```

### Output:
```
data/segments/
├── train/
│   ├── video1_0001.wav
│   ├── video1_0002.wav
│   └── ...
├── val/
└── test/
```

---

## 🔄 Full Pipeline (1 lệnh)

```bash
# Tạo script chạy toàn bộ
cd f:\Viet-Turn

# Step 1-3 (download + process + label)
python scripts/01_download_audio.py --url "YOUR_URL" --output data/raw/youtube
python scripts/02_auto_process.py --input data/raw/youtube --output data/processed/auto
python scripts/03_llm_prelabel.py --input data/processed/auto --output data/processed/labeled

# Step 4 (export for review)
python scripts/04_export_labelstudio.py --input data/processed/labeled --output data/labelstudio

# === PAUSE: Manual review on Label Studio ===

# Step 5-8 (after review)
python scripts/05_merge_reviewed.py --auto data/processed/labeled --reviewed data/labelstudio/export.json --output data/processed/final
python scripts/06_split_dataset.py --input data/processed/final --output data/final
python scripts/07_cut_segments.py --input data/final --audio-dir data/raw/youtube --output data/segments
```

---

## ⏱️ Thời Gian Ước Tính

| Step | Thời gian cho 1h audio |
|------|------------------------|
| Download | 2-5 phút |
| ASR + Diarization | 10-30 phút (GPU) |
| LLM Labeling | 5-10 phút |
| Human Review | 30-60 phút |
| Post-processing | 2-5 phút |

**Total: ~1-2 giờ / 1 giờ audio**

---

## ❓ Troubleshooting

### Lỗi HF_TOKEN
```
Cần HuggingFace token cho speaker diarization
```
→ Thêm HF_TOKEN vào file `.env`

### Lỗi GOOGLE_API_KEY
```
Cần GOOGLE_API_KEY!
```
→ Thêm GOOGLE_API_KEY vào file `.env`

### Lỗi whisperx
```
ModuleNotFoundError: No module named 'whisperx'
```
→ `pip install whisperx`

### Lỗi CUDA out of memory
→ Dùng `--model small` hoặc `--device cpu`
