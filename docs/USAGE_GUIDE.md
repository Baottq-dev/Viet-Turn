# 📖 Hướng dẫn Chạy Scripts Chi tiết - Viet-Turn Pipeline

> **Cập nhật**: 2026-01-22  
> **Thư mục scripts**: `f:\Viet-Turn\scripts`

---

## 📋 Mục lục

1. [Cài đặt Prerequisites](#1-cài-đặt-prerequisites)
2. [Script 01: Download Audio](#2-script-01-download-audio)
3. [Script 02: Auto Process (ASR + Diarization)](#3-script-02-auto-process)
4. [Script 03: LLM Pre-labeling](#4-script-03-llm-pre-labeling)
5. [Script 04: Export Label Studio](#5-script-04-export-label-studio)
6. [Script 05: Merge Reviewed](#6-script-05-merge-reviewed)
7. [Script 06: Split Dataset](#7-script-06-split-dataset)
8. [Script 07: Create Manifest](#8-script-07-create-manifest)
9. [Script 08: Extract Features](#9-script-08-extract-features)
10. [Convert SRT to JSON](#10-convert-srt-to-json)

---

## 1. Cài đặt Prerequisites

### 1.1 Cài đặt Dependencies

```powershell
cd f:\Viet-Turn
pip install -r requirements.txt
```

### 1.2 Cấu hình Environment Variables

Mở file `.env` và điền thông tin:

```env
# HuggingFace Token (bắt buộc cho diarization)
# Lấy tại: https://huggingface.co/settings/tokens
HF_TOKEN=hf_xxxxxxxxxxxxxx

# Google API Key (bắt buộc cho LLM labeling)
# Lấy tại: https://aistudio.google.com/apikey
GOOGLE_API_KEY=AIzaxxxxxxxxx
```

### 1.3 Cài đặt FFmpeg (cho yt-dlp)

```powershell
winget install Gyan.FFmpeg
# Hoặc download từ: https://ffmpeg.org/download.html
```

---

## 2. Script 01: Download Audio

**Mục đích**: Tải audio từ YouTube/Podcast

### Chạy với URL đơn lẻ

```powershell
cd f:\Viet-Turn
python scripts/01_download_audio.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --output datasets/raw/youtube
```

### Chạy với danh sách URLs

```powershell
# Tạo file urls.txt với mỗi dòng là 1 URL
python scripts/01_download_audio.py --file scripts/urls.txt --output datasets/raw/youtube
```

### Chạy với Playlist

```powershell
python scripts/01_download_audio.py --playlist "https://www.youtube.com/playlist?list=PLxxxxxx" --output datasets/raw/youtube --max-videos 10
```

### Tham số quan trọng

| Tham số | Mô tả | Default |
|---------|-------|---------|
| `--output`, `-o` | Thư mục lưu audio | `data/raw/youtube` |
| `--format` | Định dạng audio (wav, mp3, m4a) | `wav` |
| `--max-duration` | Giới hạn độ dài video (giây) | `3600` (1h) |
| `--max-videos` | Số video tối đa (playlist) | None |

---

## 3. Script 02: Auto Process

**Mục đích**: ASR (Whisper) + Diarization (Pyannote) + Overlap Detection

### ⭐ Phiên bản khuyến nghị: v2

```powershell
cd f:\Viet-Turn
python scripts/02_auto_process_v2.py --input datasets/raw/youtube --output datasets/processed/auto --enable-overlap
```

### Chạy với file đơn lẻ

```powershell
python scripts/02_auto_process_v2.py --input datasets/raw/youtube/video.wav --output datasets/processed/auto --enable-overlap
```

### Chạy trên CPU (không có GPU)

```powershell
python scripts/02_auto_process_v2.py --input datasets/raw/youtube --output datasets/processed/auto --device cpu --batch-size 4
```

### Tham số quan trọng

| Tham số | Mô tả | Default |
|---------|-------|---------|
| `--input`, `-i` | Thư mục/file audio | **Bắt buộc** |
| `--output`, `-o` | Thư mục output JSON | `data/processed/auto` |
| `--model` | Whisper model (tiny/base/small/medium/large-v3) | `large-v3` |
| `--device` | cuda hoặc cpu | `cuda` nếu có |
| `--enable-overlap` | Bật overlap detection | `False` |
| `--batch-size` | Batch size (giảm nếu CUDA OOM) | `16` |
| `--skip-diarization` | Bỏ qua speaker diarization | `False` |

### Output mẫu

```json
{
  "audio_file": "video.wav",
  "duration": 3600.0,
  "num_segments": 450,
  "num_overlaps": 23,
  "segments": [
    {
      "id": 0,
      "start": 0.5,
      "end": 3.2,
      "text": "Xin chào các bạn",
      "speaker": "SPEAKER_00",
      "has_overlap": false
    }
  ]
}
```

---

## 4. Script 03: LLM Pre-labeling

**Mục đích**: Gán nhãn YIELD/HOLD/BACKCHANNEL/INTERRUPT tự động bằng Gemini

### ⭐ Phiên bản khuyến nghị: v2 (Multimodal)

```powershell
cd f:\Viet-Turn
python scripts/03_llm_prelabel_v2.py --input datasets/processed/auto --audio-dir datasets/raw/youtube --output datasets/processed/labeled --multimodal
```

### Chạy Text-only (không cần audio)

```powershell
python scripts/03_llm_prelabel_v2.py --input datasets/processed/auto --output datasets/processed/labeled
```

### Chạy với file đơn lẻ

```powershell
python scripts/03_llm_prelabel_v2.py --input datasets/processed/auto/video.json --audio-dir datasets/raw/youtube --output datasets/processed/labeled --multimodal
```

### Tham số quan trọng

| Tham số | Mô tả | Default |
|---------|-------|---------|
| `--input`, `-i` | Thư mục/file JSON từ bước 02 | **Bắt buộc** |
| `--audio-dir`, `-a` | Thư mục chứa audio gốc | None |
| `--output`, `-o` | Thư mục output | `data/processed/labeled` |
| `--multimodal` | Gửi audio + text cho Gemini | `False` |
| `--model` | Gemini model | `gemini-1.5-flash` |
| `--text-only` | Chỉ dùng text | `False` |

### Output bổ sung

```json
{
  "segments": [
    {
      "id": 0,
      "text": "Xin chào các bạn",
      "auto_label": "YIELD",
      "confidence": 0.9,
      "intonation": "falling",
      "intensity": "normal",
      "label_mode": "multimodal",
      "needs_review": false
    }
  ],
  "label_stats": {
    "YIELD": 200,
    "HOLD": 100,
    "BACKCHANNEL": 150
  }
}
```

---

## 5. Script 04: Export Label Studio

**Mục đích**: Export dữ liệu sang Label Studio để human review

```powershell
cd f:\Viet-Turn
python scripts/04_export_labelstudio.py --input datasets/processed/labeled --output datasets/labelstudio --audio-src datasets/raw/youtube
```

### Export tất cả (không chỉ needs_review)

```powershell
python scripts/04_export_labelstudio.py --input datasets/processed/labeled --output datasets/labelstudio --all
```

### Tham số quan trọng

| Tham số | Mô tả | Default |
|---------|-------|---------|
| `--input`, `-i` | Thư mục labeled JSON | **Bắt buộc** |
| `--output`, `-o` | Thư mục output | `data/labelstudio` |
| `--audio-src` | Thư mục audio (sẽ copy) | None |
| `--all` | Export tất cả, không chỉ needs_review | `False` |
| `--threshold` | Confidence threshold | `0.7` |

### Sau khi export

1. Start Label Studio:
   ```powershell
   pip install label-studio
   label-studio start
   ```

2. Tạo project mới, import `tasks.json`

3. Setup labeling interface với `labeling_config.xml`

---

## 6. Script 05: Merge Reviewed

**Mục đích**: Merge labels đã review từ Label Studio với auto labels

```powershell
cd f:\Viet-Turn
python scripts/05_merge_reviewed.py --auto datasets/processed/labeled --reviewed datasets/labelstudio/export.json --output datasets/processed/final --validate
```

### Tham số quan trọng

| Tham số | Mô tả | Default |
|---------|-------|---------|
| `--auto`, `-a` | Thư mục auto-labeled JSON | **Bắt buộc** |
| `--reviewed`, `-r` | Label Studio export JSON | **Bắt buộc** |
| `--output`, `-o` | Thư mục output | `data/processed/final` |
| `--validate` | Chạy validation | `False` |

---

## 7. Script 06: Split Dataset

**Mục đích**: Chia dataset thành train/val/test

```powershell
cd f:\Viet-Turn
python scripts/06_split_dataset.py --input datasets/processed/final --output datasets/final --train-ratio 0.8 --val-ratio 0.1
```

### Extract features trong lúc split

```powershell
python scripts/06_split_dataset.py --input datasets/processed/final --output datasets/final --extract-features --audio-dir datasets/raw/youtube
```

### Tham số quan trọng

| Tham số | Mô tả | Default |
|---------|-------|---------|
| `--input`, `-i` | Thư mục merged data | **Bắt buộc** |
| `--output`, `-o` | Thư mục output | `data/final` |
| `--train-ratio` | Tỷ lệ train | `0.8` |
| `--val-ratio` | Tỷ lệ validation | `0.1` |
| `--seed` | Random seed | `42` |

### Output

```
datasets/final/
├── train.json
├── val.json
├── test.json
└── manifest.json
```

---

## 8. Script 07: Create Manifest

**Mục đích**: Tạo manifest cho VAP training (THAY THẾ cắt đoạn)

```powershell
cd f:\Viet-Turn
python scripts/07_create_manifest.py --input datasets/final --audio-dir datasets/raw/youtube --output datasets/manifest
```

### Tùy chỉnh window size

```powershell
python scripts/07_create_manifest.py --input datasets/final --audio-dir datasets/raw/youtube --output datasets/manifest --history-window 15.0 --prediction-window 3.0
```

### Tham số quan trọng

| Tham số | Mô tả | Default |
|---------|-------|---------|
| `--input`, `-i` | Thư mục chứa train/val/test.json | **Bắt buộc** |
| `--audio-dir`, `-a` | Thư mục audio gốc | **Bắt buộc** |
| `--output`, `-o` | Thư mục output manifest | `data/manifest` |
| `--history-window` | Cửa sổ lịch sử (giây) | `10.0` |
| `--prediction-window` | Cửa sổ dự đoán (giây) | `2.0` |

### Output

```
datasets/manifest/
├── train_manifest.json    # Combined manifest cho DataLoader
├── val_manifest.json
├── test_manifest.json
├── train/                  # Individual manifests
│   ├── video1.json
│   └── video2.json
└── ...
```

---

## 9. Script 08: Extract Features

**Mục đích**: Trích xuất F0, intensity, prosodic features

```powershell
cd f:\Viet-Turn
python scripts/08_extract_features.py --input datasets/final --audio-dir datasets/raw/youtube --output datasets/features
```

### Export JSON thay vì PyTorch

```powershell
python scripts/08_extract_features.py --input datasets/final --audio-dir datasets/raw/youtube --output datasets/features --format json
```

### Tham số quan trọng

| Tham số | Mô tả | Default |
|---------|-------|---------|
| `--input`, `-i` | Thư mục chứa train/val/test.json | **Bắt buộc** |
| `--audio-dir`, `-a` | Thư mục audio gốc | **Bắt buộc** |
| `--output`, `-o` | Thư mục output features | `data/features` |
| `--format` | Output format (pt, json) | `pt` |
| `--f0-min` | F0 minimum (Hz) | `75.0` |
| `--f0-max` | F0 maximum (Hz) | `500.0` |

### Output

```
datasets/features/
├── train_features.pt    # PyTorch tensor
├── val_features.pt
└── test_features.pt
```

### Features bao gồm

| Feature | Mô tả |
|---------|-------|
| f0_mean, f0_std, f0_range | Thống kê F0 |
| f0_slope | Hướng ngữ điệu (rising/falling) |
| f0_final | F0 cuối segment (quan trọng cho turn-taking) |
| intensity_mean, intensity_std | Cường độ giọng |
| *_zscore | Z-score normalized theo speaker |

---

## 10. Convert SRT to JSON

**Mục đích**: Chuyển file phụ đề SRT thành JSON pipeline format

```powershell
cd f:\Viet-Turn
python scripts/convert_srt_to_json.py --input datasets/dataset-youtube-sub/sub --output datasets/processed/srt
```

### Không merge segments ngắn

```powershell
python scripts/convert_srt_to_json.py --input datasets/dataset-youtube-sub/sub --output datasets/processed/srt --no-merge
```

---

## 🔄 Full Pipeline Example

```powershell
cd f:\Viet-Turn

# 1. Download audio
python scripts/01_download_audio.py --file scripts/urls.txt --output datasets/raw/youtube

# 2. ASR + Diarization + Overlap Detection
python scripts/02_auto_process_v2.py --input datasets/raw/youtube --output datasets/processed/auto --enable-overlap

# 3. LLM Labeling (Multimodal)
python scripts/03_llm_prelabel_v2.py --input datasets/processed/auto --audio-dir datasets/raw/youtube --output datasets/processed/labeled --multimodal

# 4. Export to Label Studio (optional)
python scripts/04_export_labelstudio.py --input datasets/processed/labeled --output datasets/labelstudio

# 5. Merge reviewed (sau khi review xong)
python scripts/05_merge_reviewed.py --auto datasets/processed/labeled --reviewed datasets/labelstudio/export.json --output datasets/processed/final

# 6. Split dataset
python scripts/06_split_dataset.py --input datasets/processed/final --output datasets/final

# 7. Create manifest (for VAP training)
python scripts/07_create_manifest.py --input datasets/final --audio-dir datasets/raw/youtube --output datasets/manifest

# 8. Extract features
python scripts/08_extract_features.py --input datasets/final --audio-dir datasets/raw/youtube --output datasets/features
```

---

## ❓ Troubleshooting

### CUDA Out of Memory

```powershell
# Giảm batch size
python scripts/02_auto_process_v2.py ... --batch-size 4
```

### Missing HF_TOKEN

```powershell
# Set trong terminal
$env:HF_TOKEN = "hf_xxxxxx"
# Hoặc thêm vào .env
```

### Missing GOOGLE_API_KEY

```powershell
$env:GOOGLE_API_KEY = "AIzaxxxxxx"
```

### Audio file not found

Kiểm tra tên file trong JSON khớp với tên file thực tế trong thư mục audio.

---

## 📊 Thư mục Structure sau khi chạy

```
f:\Viet-Turn\
├── datasets/
│   ├── raw/
│   │   └── youtube/           # Audio gốc (.wav)
│   ├── processed/
│   │   ├── auto/             # ASR output
│   │   ├── labeled/          # LLM labeled
│   │   └── final/            # Merged & reviewed
│   ├── final/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   ├── manifest/              # VAP manifest
│   └── features/              # Prosodic features
├── scripts/
└── .env
```
