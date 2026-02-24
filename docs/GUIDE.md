# MM-VAP-VI: Huong dan chay tu dau den cuoi

## Muc luc

1. [Tong quan tinh trang Implementation](#1-tong-quan-tinh-trang-implementation)
2. [Cai dat moi truong](#2-cai-dat-moi-truong)
3. [Chuan bi du lieu](#3-chuan-bi-du-lieu)
4. [Training](#4-training)
5. [Evaluation](#5-evaluation)
6. [Streaming Inference](#6-streaming-inference)
7. [Chay Tests](#7-chay-tests)
8. [Danh gia GPU RTX 4060](#8-danh-gia-gpu-rtx-4060)
9. [Gap Analysis: Doc vs Code](#9-gap-analysis-doc-vs-code)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Tong quan tinh trang Implementation

### Da implement hoan thien

| Component | File(s) | Trang thai |
|-----------|---------|------------|
| Acoustic Encoder (Wav2Vec2/WavLM) | `src/models/acoustic_encoder.py` | Done |
| Linguistic Encoder (PhoBERT + HuTuDetector) | `src/models/linguistic_encoder.py` | Done |
| Cross-Attention / GMU / Bottleneck Fusion | `src/models/fusion.py` | Done (3 strategies) |
| Causal Transformer + ALiBi + KV Cache | `src/models/transformer.py` | Done |
| VAP Projection Head (256 classes) | `src/models/projection_head.py` | Done |
| Full Model Orchestrator | `src/models/model.py` | Done |
| VAPDataset (sliding window) | `src/data/dataset.py` | Done |
| Collate Function | `src/data/collate.py` | Done |
| 3-Stage Trainer | `src/training/trainer.py` | Done |
| VAPLoss (CE/Focal + transition weight + label smoothing) | `src/training/losses.py` | Done |
| Augmentation (modality dropout, time mask, noise) | `src/training/augmentation.py` | Done |
| VAP Label Encode/Decode | `src/utils/labels.py` | Done |
| Audio Utilities | `src/utils/audio.py` | Done |
| Evaluation: Tier 1 Frame Metrics | `src/evaluation/frame_metrics.py` | Done |
| Evaluation: Tier 2 Event Metrics | `src/evaluation/event_metrics.py` | Done |
| Evaluation: Tier 3 Latency + VAQI + EoT Levenshtein | `src/evaluation/latency_metrics.py` | Done |
| Evaluation: Statistical Tests (permutation test) | `src/evaluation/statistical.py` | Done |
| Evaluation: Vietnamese Analysis (marker/dialect) | `src/evaluation/vietnamese_analysis.py` | Done |
| Evaluation: Orchestrator (4-tier) | `src/evaluation/evaluator.py` | Done |
| Streaming Inference (KV cache) | `src/inference/streaming.py` | Done |
| Data Pipeline Scripts (00-07) | `scripts/` | Done (9 scripts) |
| Unit Tests | `tests/` | Done (4 test files) |
| Training Entry Point | `train.py` | Done |

### Chua implement (optional, khong can cho V1)

| Component | Trang thai | Ghi chu |
|-----------|------------|---------|
| Tonal F0 Analysis | **Chua implement** | Analysis tool, lam sau khi co ket qua training |
| Macro F1 | **Chua implement** | Chi co weighted F1, them 1 dong khi can |
| Time Stretch Augmentation | **Chua implement** | 3 augmentation hien tai du cho V1 |
| Whisper Encoder option | **Chua implement** | Wav2Vec2-VI da la best choice cho Vietnamese |
| Stereo/Dual-channel Encoder | **Khac voi docs** | Code dung mono — **dung** vi input la mono + diarization |

### Nhan xet tong the

Code **khong co bat ky TODO, FIXME, hay NotImplementedError nao**. Pipeline co the chay end-to-end tu raw audio den training, evaluation (bao gom VAQI, marker analysis), va streaming inference. Cac item con lai deu la optional features, khong anh huong core pipeline.

---

## 2. Cai dat moi truong

### 2.1 Yeu cau he thong

- Python >= 3.10
- CUDA >= 11.8 (cho PyTorch 2.x)
- GPU >= 8GB VRAM (RTX 4060 co 8GB - xem danh gia chi tiet o Section 8)
- Disk: ~20GB cho models pretrained, ~50-100GB cho dataset

### 2.2 Setup

```bash
# Clone project
git clone <repo-url> Viet-Turn
cd Viet-Turn

# Tao virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoac: venv\Scripts\activate  # Windows

# Cai dependencies (THU TU QUAN TRONG!)
# Buoc 1: PyTorch 2.5.1 voi CUDA 12.4 (tuong thich pyannote 3.x)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# Buoc 2: Cai cac package con lai
pip install -r requirements.txt

# Config environment variables
cp .env.example .env
# Sua .env: dien HF_TOKEN va WANDB_API_KEY (optional)
```

### 2.3 HuggingFace Token

Can HF Token de:
1. Download `pyannote/speaker-diarization-3.1` (can accept license tren HF)
2. Download `nguyenvulebinh/wav2vec2-base-vi`
3. Download `vinai/phobert-base-v2`

```bash
# Login HuggingFace (hoac dien vao .env)
huggingface-cli login
```

### 2.4 Kiem tra cai dat

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "from transformers import AutoModel; print('transformers OK')"
python -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"
```

---

## 3. Chuan bi du lieu

### 3.1 Cau truc thu muc du lieu

```
data/
├── audio/              # Raw audio files (.wav, 16kHz, mono)
├── audio_split/        # Split segments (~10 min each, .wav)
├── rttm/               # Speaker diarization output
├── va_matrices/        # Binary voice activity matrices (.pt)
├── vap_labels/         # 256-class VAP labels (.pt)
├── transcripts/        # ASR word-level transcripts (.json)
├── text_frames/        # Frame-aligned text snapshots (.json)
├── vap_manifest_train.json
├── vap_manifest_val.json
└── vap_manifest_test.json
```

### 3.2 Chuan bi audio

Dat file audio vao `data/audio/`. Yeu cau:
- Format: WAV, 16kHz, mono
- Do dai: 20-60 phut moi file (conversation/podcast)
- Noi dung: hoi thoai 2 nguoi (Vietnamese)

**Download tu YouTube** (da co san 10 link trong `scripts/urls.txt`):

```bash
# === Buoc 0: Download audio tu YouTube ===
# Download tat ca video, tu dong convert WAV 16kHz mono (~10-30 phut tuy mang)
python scripts/00_download_audio.py

# Hoac voi options:
python scripts/00_download_audio.py --urls scripts/urls.txt --output data/audio
python scripts/00_download_audio.py --max-duration 3600   # bo qua video > 1h
python scripts/00_download_audio.py --dry-run              # xem truoc, khong download
python scripts/00_download_audio.py --verify               # kiem tra file da download
```

File `scripts/urls.txt` chua 10 video tu nhieu kenh/the loai:
- **Bar Stories** (Dustin On The Go) - entertainment talkshow
- **Have A Sip** (Vietcetera) - lifestyle/culture
- **Vietnam Innovators** (Vietcetera) - business/tech
- **Khong Cay Khong Ve** (Vietcetera) - food/culinary
- **Coi Mo** (Vietcetera) - social topics

**Cat audio dai thanh ~10 phut** (podcast 30-90 phut → segments ~10 phut, dung Silero VAD):

```bash
# === Buoc 0b: Split audio dai thanh segments ngan ===
# Dung Silero VAD (neural network) detect pause giua loi noi, cat tai pause dai nhat
python scripts/00b_split_audio.py --input data/audio --output data/audio_split

# Xem truoc split plan, khong ghi file:
python scripts/00b_split_audio.py --input data/audio --output data/audio_split --dry-run

# Custom: target 15 phut/segment, chi cat tai pause >= 0.5s
python scripts/00b_split_audio.py --input data/audio --output data/audio_split --segment-min 15 --min-pause 0.5
```

Sau khi split, dung `data/audio_split` lam input cho cac buoc tiep theo thay vi `data/audio`.

Neu co audio san (khong download tu YouTube):

```bash
# Convert sang dung format:
ffmpeg -i input.mp3 -ar 16000 -ac 1 data/audio/output.wav
```

### 3.3 Chay pipeline tu dong (8 buoc)

```bash
# === Buoc 0b: Split Audio (neu file > 15 phut) ===
# Silero VAD detect speech pauses, cat tai pause dai nhat gan moc ~10 phut
python scripts/00b_split_audio.py \
    --input data/audio \
    --output data/audio_split \
    --segment-min 10
# Luu y: Cac buoc sau dung data/audio_split thay vi data/audio

# === Buoc 1: Speaker Diarization ===
# Phan biet nguoi noi trong audio (can GPU, ~2-5 phut/file, ~2GB VRAM)
python scripts/01_diarize.py \
    --input data/audio \
    --output data/rttm \
    --num-speakers 2

# === Buoc 2: Build Voice Activity Matrix ===
# Chuyen RTTM -> binary matrix (2, num_frames) tai 50fps (CPU only, nhanh)
python scripts/02_build_va_matrix.py \
    --rttm-dir data/rttm \
    --audio-dir data/audio \
    --output data/va_matrices

# === Buoc 3: Generate VAP Labels ===
# Chuyen VA matrix -> 256-class labels (self-supervised, CPU only, nhanh)
python scripts/03_generate_labels.py \
    --input data/va_matrices \
    --output data/vap_labels

# === Buoc 4: ASR Transcription ===
# Tao transcript voi word-level timestamps (can GPU, ~5-10 phut/file, ~3GB VRAM)
python scripts/04_transcribe.py \
    --input data/audio \
    --output data/transcripts \
    --model large-v3

# === Buoc 5: Text-Frame Alignment ===
# Map word timestamps -> 50fps frame indices (CPU only, nhanh)
python scripts/05_align_text.py \
    --transcripts data/transcripts \
    --va-matrices data/va_matrices \
    --output data/text_frames

# === Buoc 6: Create Manifest ===
# Tao train/val/test manifest JSON (80/10/10 split)
python scripts/06_create_manifest.py \
    --audio-dir data/audio \
    --va-dir data/va_matrices \
    --label-dir data/vap_labels \
    --text-dir data/text_frames \
    --output data

# === Buoc 7: Validate Data ===
# Kiem tra chat luong dataset
python scripts/07_validate_data.py \
    --manifest data/vap_manifest_train.json
```

### 3.4 Script tat-ca-trong-mot

```bash
#!/bin/bash
# run_pipeline.sh - Chay toan bo pipeline
set -e

AUDIO_RAW="data/audio"
AUDIO_DIR="data/audio_split"
RTTM_DIR="data/rttm"
VA_DIR="data/va_matrices"
LABEL_DIR="data/vap_labels"
TRANSCRIPT_DIR="data/transcripts"
TEXT_DIR="data/text_frames"
OUTPUT_DIR="data"

echo "=== Step 0: Download Audio ==="
python scripts/00_download_audio.py --output $AUDIO_RAW

echo "=== Step 0b: Split Long Audio ==="
python scripts/00b_split_audio.py --input $AUDIO_RAW --output $AUDIO_DIR --segment-min 10

echo "=== Step 1: Diarization ==="
python scripts/01_diarize.py --input $AUDIO_DIR --output $RTTM_DIR --num-speakers 2

echo "=== Step 2: VA Matrix ==="
python scripts/02_build_va_matrix.py --rttm-dir $RTTM_DIR --audio-dir $AUDIO_DIR --output $VA_DIR

echo "=== Step 3: VAP Labels ==="
python scripts/03_generate_labels.py --input $VA_DIR --output $LABEL_DIR

echo "=== Step 4: Transcription ==="
python scripts/04_transcribe.py --input $AUDIO_DIR --output $TRANSCRIPT_DIR --model large-v3

echo "=== Step 5: Text Alignment ==="
python scripts/05_align_text.py --transcripts $TRANSCRIPT_DIR --va-matrices $VA_DIR --output $TEXT_DIR

echo "=== Step 6: Create Manifest ==="
python scripts/06_create_manifest.py --audio-dir $AUDIO_DIR --va-dir $VA_DIR --label-dir $LABEL_DIR --text-dir $TEXT_DIR --output $OUTPUT_DIR

echo "=== Step 7: Validate ==="
python scripts/07_validate_data.py --manifest ${OUTPUT_DIR}/vap_manifest_train.json

echo "=== Pipeline hoan tat! ==="
```

---

## 4. Training

### 4.1 Chay training (3-stage)

```bash
# Training day du 3 stage (Stage 1: 10 epochs, Stage 2: 20, Stage 3: 20)
python train.py --config configs/config.yaml --device cuda

# Voi custom output directory
python train.py --config configs/config.yaml --output-dir outputs/exp_01

# Resume tu checkpoint
python train.py --config configs/config.yaml --resume outputs/mm_vap/checkpoint_s2_e5.pt
```

### 4.2 Cac stage training

| Stage | Epochs | Trainable | Mo ta |
|-------|--------|-----------|-------|
| 1 | 10 | Acoustic + Transformer + Head | Audio-only, freeze text + fusion |
| 2 | 20 | + Fusion + last 2 PhoBERT layers + HuTu | Multimodal, them text |
| 3 | 20 | All (tru CNN extractor) | Full fine-tune |

### 4.3 Theo doi training

```bash
# Wandb (neu da cau hinh)
# Truy cap: https://wandb.ai/<entity>/mm-vap-vi

# Hoac xem history file
cat outputs/mm_vap/training_history.json | python -m json.tool
```

### 4.4 Config quan trong cho RTX 4060

Sua `configs/config.yaml` de fit GPU 8GB:

```yaml
training:
  batch_size: 2          # Giam tu 4 xuong 2
  accumulate_grad_batches: 8  # Tang len 8 de giu effective batch = 16
  mixed_precision: true  # BAT BUOC fp16
```

---

## 5. Evaluation

### 5.1 Chay evaluation

```python
# evaluation_script.py
import torch
from src.models.model import MMVAPModel
from src.data.dataset import VAPDataset
from src.data.collate import vap_collate_fn
from src.evaluation.evaluator import MMVAPEvaluator
from torch.utils.data import DataLoader

# Load model
model = MMVAPModel.from_config("configs/config.yaml")
ckpt = torch.load("outputs/mm_vap/best_model.pt", map_location="cuda")
model.load_state_dict(ckpt["model_state_dict"])

# Load test data
test_dataset = VAPDataset(
    manifest_path="data/vap_manifest_test.json",
    window_sec=20.0, stride_sec=5.0,
    frame_hz=50, sample_rate=16000,
)
test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=vap_collate_fn)

# Evaluate
evaluator = MMVAPEvaluator(model, test_loader, device="cuda")
results = evaluator.evaluate()
evaluator.print_report(results)
evaluator.save_report(results, "outputs/eval_report.json")
```

### 5.2 Metrics gom

**Tier 1 - Frame Level**: CE Loss, Perplexity, Top-1/5 Accuracy, Weighted F1, ECE, Brier Score

**Tier 2 - Event Level**: Shift/Hold Balanced Accuracy, Backchannel F1, Predict-Shift AUC, 3-class BA

**Tier 3 - Latency**: EOT Latency (target <200ms), FPR (target <5%), MST-FPR curve

**Tier 4 - Aggregate** (VAQI + marker analysis da tu dong chay trong `evaluator.evaluate()`. Cac utility khac goi truc tiep):

```python
from src.evaluation.latency_metrics import compute_vaqi, compute_eot_levenshtein
from src.evaluation.statistical import permutation_test
from src.evaluation.vietnamese_analysis import analyze_marker_impact, analyze_per_dialect

# VAQI - Voice Agent Quality Index
vaqi = compute_vaqi(p_shift, gt_shift_regions, gt_hold_regions)
print(f"VAQI: {vaqi['vaqi']}/100")

# EoT Levenshtein alignment
eot = compute_eot_levenshtein(gt_eot_frames, pred_eot_frames, tolerance_frames=25)
print(f"EoT F1: {eot['eot_f1']:.4f}")

# Ablation comparison (paired permutation test)
result = permutation_test(scores_model_a, scores_model_b, n_permutations=10000)
print(f"p-value: {result['p_value']}, significant: {result['significant_at_05']}")

# Vietnamese discourse marker impact
impact = analyze_marker_impact(events, p_shift)
print(f"Marker benefit (shift): {impact['marker_benefit']['shift_delta']:+.4f}")

# Per-dialect breakdown
dialect_results = analyze_per_dialect(per_conversation_results)
for dialect, metrics in dialect_results.items():
    print(f"{dialect}: BA={metrics['shift_hold_ba']:.4f}")
```

---

## 6. Streaming Inference

```python
import torch
import torchaudio
from src.models.model import MMVAPModel
from src.inference.streaming import StreamingMMVAP

# Load model
model = MMVAPModel.from_config("configs/config.yaml")
ckpt = torch.load("outputs/mm_vap/best_model.pt", map_location="cuda")
model.load_state_dict(ckpt["model_state_dict"])

# Tao streaming wrapper (KV cache enabled by default)
streamer = StreamingMMVAP(
    model=model,
    device="cuda",
    chunk_ms=200,               # Process moi 200ms (10 frames)
    max_context_frames=1000,    # 20 sec context
    use_kv_cache=True,          # Reuse past KV (tang toc ~3-5x)
)

# Process audio chunks (simulate real-time)
audio, sr = torchaudio.load("test_audio.wav")
chunk_samples = int(0.2 * sr)  # 200ms = 3200 samples at 16kHz

for i in range(0, audio.shape[1], chunk_samples):
    chunk = audio[0, i:i+chunk_samples]
    text = "current transcript so far..."  # tu ASR streaming

    result = streamer.process_audio(chunk, text=text)
    if result is not None:
        probs = result["probs"]       # (num_new_frames, 256)
        p_now = result["p_now"]       # (num_new_frames, 2) per-speaker
        events = result["events"]     # List of detected events
        print(f"Frame {result['frame_offset']}: "
              f"p_now=[{p_now[-1, 0]:.2f}, {p_now[-1, 1]:.2f}], "
              f"events={len(events)}")

# Kiem tra trang thai streaming
print(streamer.get_current_state())

# Reset cho conversation moi
streamer.reset()
```

**Luu y ve KV cache**:
- KV cache tu dong invalidate khi text thay doi (goi `update_text()`)
- KV cache tu dong invalidate khi audio buffer bi trim (vuot `max_context_frames`)
- Tat KV cache bang `use_kv_cache=False` neu muon full forward moi chunk

---

## 7. Chay Tests

```bash
# Chay tat ca tests
pytest tests/ -v

# Chay tung test file
pytest tests/test_labels.py -v      # Test VAP label encode/decode
pytest tests/test_metrics.py -v     # Test evaluation metrics
pytest tests/test_fusion.py -v      # Test 3 fusion strategies
pytest tests/test_augmentation.py -v # Test augmentation pipeline

# Chay voi coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 8. Danh gia GPU RTX 4060

### 8.1 Thong so RTX 4060

| Thong so | Gia tri |
|----------|---------|
| VRAM | 8 GB GDDR6 |
| CUDA Cores | 3072 |
| Tensor Cores | 96 (Gen 4) |
| Memory Bandwidth | 272 GB/s |
| FP16 Performance | ~15.1 TFLOPS |
| TDP | 115W |

### 8.2 Uoc tinh VRAM

**Model parameters** (~237M total, nhung da co frozen):

| Component | Parameters | VRAM (fp16) |
|-----------|-----------|-------------|
| Wav2Vec2-VI (95M, phan lon frozen) | 95M | ~190 MB |
| PhoBERT (135M, phan lon frozen) | 135M | ~270 MB |
| Fusion + Transformer + Head | ~7M | ~14 MB |
| **Model total** | **~237M** | **~474 MB** |

**Activation memory** (batch_size=2, window=20s):

| Component | Estimation |
|-----------|-----------|
| Audio waveform: (2, 320000) fp32 | ~2.4 MB |
| Wav2Vec2 activations: (2, 1000, 768) | ~6 MB |
| PhoBERT activations: (2, 256, 768) | ~1.5 MB |
| Transformer (4 layers, gradient checkpointing): (2, 1000, 256) | ~4 MB |
| Gradient buffer | ~474 MB |
| Optimizer states (AdamW, 2 moments) | ~56 MB (chi trainable params) |
| **Total activation + gradient** | **~544 MB** |

**Tong VRAM uoc tinh voi batch_size=2, fp16**:

```
Model:        ~474 MB
Activations:  ~544 MB
CUDA overhead: ~500 MB
Wav2Vec2 intermediate (peak): ~1.5 GB
PhoBERT intermediate (peak):  ~0.8 GB
─────────────────────────────
Total peak:   ~3.8 GB
```

### 8.3 Ket luan: RTX 4060 CO THE TRAIN DUOC

| Cau hinh | batch_size | accumulate | Effective batch | VRAM uoc tinh | Kha thi? |
|----------|-----------|------------|-----------------|---------------|----------|
| Config goc | 4 | 4 | 16 | ~6.5 GB | Sat gioi han |
| **Khuyen nghi** | **2** | **8** | **16** | **~3.8 GB** | **An toan** |
| Toi thieu | 1 | 16 | 16 | ~2.8 GB | Du thua |

### 8.4 Config khuyen nghi cho RTX 4060

Sua `configs/config.yaml`:

```yaml
# === Thay doi cho RTX 4060 (8GB) ===

acoustic_encoder:
  gradient_checkpointing: true    # Giu nguyen - tiet kiem ~40% VRAM

transformer:
  gradient_checkpointing: true    # Giu nguyen

training:
  batch_size: 2                   # Giam tu 4 -> 2
  accumulate_grad_batches: 8      # Tang tu 4 -> 8 (giu effective batch = 16)
  mixed_precision: true           # BAT BUOC - giam VRAM ~50%
```

### 8.5 Uoc tinh thoi gian training

Gia su dataset ~50 gio audio (~9000 sliding windows @ 20s/5s stride):

| Stage | Epochs | Batches/epoch | Time/batch (est.) | Time/stage |
|-------|--------|--------------|-------------------|------------|
| Stage 1 | 10 | 4500 | ~0.3s | ~3.7h |
| Stage 2 | 20 | 4500 | ~0.5s | ~12.5h |
| Stage 3 | 20 | 4500 | ~0.6s | ~15h |
| **Total** | **50** | | | **~31h** |

Voi early stopping (patience=5), thuc te co the ngan hon: **~20-25 gio**.

### 8.6 Meo toi uu cho 4060

1. **Tat wandb logging neu khong can**: bot I/O overhead
2. **Giam `num_workers`**: `--num-workers 2` thay vi 4 (tranh bottleneck CPU-GPU)
3. **Pre-process audio sang .pt**: Load nhanh hon `.wav`
4. **Monitor VRAM**: Dung `nvidia-smi -l 1` de theo doi
5. **Neu OOM**: Giam `batch_size` xuong 1, tang `accumulate_grad_batches` len 16

---

## 9. Gap Analysis: Doc vs Code

### DA RESOLVE

| # | Van de | Trang thai |
|---|--------|------------|
| 1 | Projection Head thieu Dropout | **DA FIX** — `Dropout(0.1)` da them |
| 2 | VAQI chua wire vao evaluator | **DA FIX** — `evaluator.evaluate()` goi `compute_vaqi()` tu dong |
| 3 | Vietnamese marker analysis chua wire | **DA FIX** — `evaluator.evaluate()` goi `analyze_marker_impact()` tu dong |

### CON LAI (khong anh huong core pipeline)

| # | Van de | Trang thai | Ghi chu |
|---|--------|------------|---------|
| 4 | Acoustic encoder mono vs stereo | Docs sai | Code dung mono — **dung** vi input la mono + diarization |
| 5 | Tonal F0 analysis | Chua implement | Analysis tool, lam sau khi co ket qua training |
| 6 | Time stretch augmentation | Chua implement | 3 augmentation hien tai du cho V1 |
| 7 | Whisper encoder option | Chua implement | Wav2Vec2-VI da la best choice |
| 8 | PhoBERT pooling chi co [CLS] | Chi 1 option | [CLS] la standard, mean/last-K la optional |

### MINOR - Khac biet config

| # | Parameter | Docs | Config | Ghi chu |
|---|-----------|------|--------|---------|
| 9 | Acoustic freeze_layers | 6 | 8 | Config la truth, 8 layers freeze nhieu hon = an toan hon |
| 10 | PhoBERT freeze_layers | 6 | 8 | Tuong tu |
| 11 | Fusion num_heads | 8 | 4 | 4 heads tiet kiem hon, du cho 256-dim |

### Ket luan

**Project da implement ~95% so voi design docs.** Core pipeline hoan chinh: data -> model -> training -> evaluation (bao gom VAQI + marker analysis) -> streaming inference (voi KV cache). Cac gap con lai deu la optional features, khong anh huong training hay paper submission.

Pipeline san sang de train va evaluate. RTX 4060 du kha nang voi config da dieu chinh.

---

## 10. Troubleshooting

### CUDA Out of Memory (OOM)

```bash
# Giam batch_size
python train.py --config configs/config.yaml  # sua batch_size: 1 trong config

# Kiem tra VRAM usage
nvidia-smi -l 1
```

Neu van OOM:
- Dam bao `mixed_precision: true` trong config
- Dam bao `gradient_checkpointing: true` cho ca acoustic_encoder va transformer
- Giam `window_sec` tu 20.0 xuong 10.0 (ngan hon nhung it context hon)

### HuggingFace Token Error

```
OSError: You need to accept the license for pyannote/speaker-diarization-3.1
```

Fix: Truy cap https://huggingface.co/pyannote/speaker-diarization-3.1, accept license, roi:
```bash
huggingface-cli login
```

### ffmpeg not found

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (dung chocolatey)
choco install ffmpeg

# Mac
brew install ffmpeg
```

### Import Error sau khi doi ten file

Neu gap `ModuleNotFoundError: No module named 'src.vap'`:
- Project da chuyen tu `src.vap.*` sang `src.*` (flat structure)
- Kiem tra tat ca import dung `from src.models.xxx import ...` (khong co `.vap.`)

### Diarization output xau

Neu pyannote cho ket qua sai:
- Thu chi dinh so nguoi noi: `--num-speakers 2`
- Kiem tra audio quality: mono, 16kHz, khong co nhieu nen qua lon
- Neu file dai >1h, xem xet cat thanh nhieu file ngan hon

### Tieng Viet bi loi ky tu tren Windows (PowerShell/CMD)

Khi chay script, tieng Viet hien thi sai kieu `Thành Lc` thay vi `Thành Lộc`:

```powershell
# Fix 1: Set code page UTF-8 truoc khi chay (chay 1 lan moi khi mo terminal)
chcp 65001

# Fix 2: Hoac set bien moi truong (vinh vien)
[System.Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "User")

# Fix 3: Hoac them vao PowerShell profile (tu dong moi lan mo terminal)
# Mo file profile:
notepad $PROFILE
# Them dong nay vao cuoi file:
# $OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

Luu y: Loi nay chi anh huong **hien thi tren terminal**, khong anh huong file output (file ten la video ID, vd: `PSGO2CFhBM8.wav`).
