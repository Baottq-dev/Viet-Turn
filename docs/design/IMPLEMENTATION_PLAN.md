# MM-VAP-VI Implementation Plan

## Context
The Viet-Turn project currently implements **turn classification** (segment-level YIELD/HOLD/BACKCHANNEL on completed turns). The goal is to migrate to **Voice Activity Projection (VAP)** — a frame-level, self-supervised approach that predicts future voice activity in real-time, enabling end-of-turn prediction for reducing latency in human-AI conversations. This is a **research project** targeting SOTA and paper publication.

All source code lives under `src/`, with configs at top-level `configs/`.

---

## Phase 0: Foundation & Utilities (2 days)

### 0.1 Project structure
Create directory tree:
```
src/
├── __init__.py
├── utils/
│   ├── __init__.py
│   ├── labels.py      # VAP encode/decode (256-class ↔ binary matrix)
│   └── audio.py     # Wav2Vec2/WavLM feature extraction helpers
├── data/
│   ├── __init__.py
│   ├── dataset.py          # VAPDataset (sliding window)
│   └── collate.py          # Collate function with padding
├── models/
│   ├── __init__.py
│   ├── acoustic_encoder.py # Wav2Vec2/WavLM wrapper
│   ├── linguistic_encoder.py # PhoBERT + HuTuDetector
│   ├── fusion.py           # Cross-Attention fusion
│   ├── transformer.py      # Causal Transformer + ALiBi
│   ├── projection_head.py  # 256-class VAP head
│   └── model.py           # Full MM-VAP-VI model
├── training/
│   ├── __init__.py
│   ├── trainer.py          # VAPTrainer (3-stage)
│   └── losses.py           # Frame-level CE + optional focal
├── evaluation/
│   ├── __init__.py
│   ├── frame_metrics.py    # Frame-level accuracy, BCE
│   ├── event_metrics.py    # Shift/Hold BA, BC F1
│   ├── latency_metrics.py  # EOT latency, FPR
│   └── evaluator.py        # Orchestrator
configs/
└── config.yaml              # All hyperparameters
```

### 0.2 VAP label utilities — `src/utils/labels.py`
- `encode_vap_labels(va_matrix, bins=[20,40,60,80])` → 256-class indices per frame
- `decode_vap_labels(class_idx)` → binary voice activity predictions
- `vap_to_events(probs, threshold)` → Shift/Hold/Backchannel events
- Bins at 50fps: [0-200ms, 200-600ms, 600-1200ms, 1200-2000ms] → 4 bins × 2 speakers = 8 bits → 256 classes
- Reuse vectorized approach from DATASET_GUIDE.md

### 0.3 Config — `configs/config.yaml`
- All hyperparameters in one place: model dims, training stages, data paths, evaluation thresholds

**Existing code to reuse:**
- `src/models/linguistic_branch.py` — HuTuDetector marker list and detection logic
- `configs/model_config.yaml` — reference for config structure

---

## Phase 1: Data Pipeline Scripts (8 days)

### 1.1 Speaker diarization — `scripts/01_diarize.py`
- Input: raw mono WAV files
- pyannote/speaker-diarization-3.1 → RTTM files
- Confidence filtering (skip segments < threshold)
- Output: `data/rttm/{file_id}.rttm`

### 1.2 Voice activity matrix — `scripts/02_build_va_matrix.py`
- Input: RTTM files
- Build binary voice activity matrix at 50fps (20ms frames)
- Handle >2 speakers by selecting top-2 by duration
- Output: `data/va_matrices/{file_id}.pt` — tensor (2, num_frames)

### 1.3 VAP label generation — `scripts/03_generate_labels.py`
- Input: VA matrices
- Use `encode_vap_labels()` from Phase 0
- Output: `data/vap_labels/{file_id}.pt` — tensor (num_frames,) with values 0-255

### 1.4 ASR transcription — `scripts/04_transcribe.py`
- faster-whisper with word timestamps (reuse pattern from `scripts/02_llm_process.py`)
- Output: `data/transcripts/{file_id}.json` — list of {word, start, end}

### 1.5 Text-frame alignment — `scripts/05_align_text.py`
- Map word timestamps to 50fps frame indices
- Cumulative text at each frame for PhoBERT input
- Output: `data/text_frames/{file_id}.json`

### 1.6 Dataset manifest — `scripts/06_create_manifest.py`
- Combine all artifacts into manifest JSON
- Train/val/test split (80/10/10 by conversation, not by window)
- Output: `data/vap_manifest_{split}.json`

### 1.7 Quality validation — `scripts/07_validate_data.py`
- Check label distribution (expect ~70% class 0)
- Verify frame counts match across modalities
- Flag problematic files (silence >80%, overlap >50%)

**Existing code to reuse:**
- `scripts/02_llm_process.py` — faster-whisper ASR setup, audio loading patterns

---

## Phase 2: Dataset & Model (10 days)

### 2.1 VAPDataset — `src/data/dataset.py`
- Sliding window: 20s windows, 5s stride (1000 frames at 50fps)
- Returns: `{audio_waveform, text_tokens, vap_labels, va_matrix, attention_mask}`
- Audio: raw waveform (Wav2Vec2 handles feature extraction)
- Text: tokenized with PhoBERT tokenizer, aligned to frames
- Collate with padding to max length in batch

### 2.2 Acoustic Encoder — `src/models/acoustic_encoder.py`
- Wrap `Wav2Vec2Model.from_pretrained("nguyenvulebinh/wav2vec2-base-vi")` or WavLM
- Freeze first N layers (configurable), fine-tune last layers
- Linear projection: 768 → 256
- Output: (B, T, 256) at 50fps

### 2.3 Linguistic Encoder — `src/models/linguistic_encoder.py`
- PhoBERT-base-v2 with [CLS] pooling → (B, 768)
- Linear projection: 768 → 256
- HuTuDetector: reuse marker list from `src/models/linguistic_branch.py`
  - Learned embeddings (dim=64) for 35 markers
  - Exponential position weighting: `exp(-α × distance_to_end)`
  - Output: (B, 64) → project to 256
- Concatenate PhoBERT + HuTuDetector → (B, 512) → Linear → (B, 256)
- Broadcast to time: (B, 256) → (B, T, 256)

### 2.4 Cross-Attention Fusion — `src/models/fusion.py`
- Audio queries, text keys/values (and vice versa)
- Causal mask on time dimension
- 4 heads, dim=256
- Residual + LayerNorm
- Output: (B, T, 256)

### 2.5 Causal Transformer — `src/models/transformer.py`
- 4 layers, 8 heads, dim=256, FFN=1024
- ALiBi positional encoding (no learned positions)
- Causal attention mask (each frame sees only past + current)
- Output: (B, T, 256)

### 2.6 VAP Projection Head — `src/models/projection_head.py`
- Linear(256, 256) → GELU → Linear(256, 256_classes)
- Output: (B, T, 256) logits

### 2.7 Full Model — `src/models/model.py`
- `MMVAPModel`: wires all components together
- Forward: audio_waveform + text_tokens → (B, T, 256) logits
- `from_config(yaml_path)` class method
- Gradient checkpointing support on transformer layers
- Parameter count target: ~120M total

**Existing code to reuse:**
- `src/models/linguistic_branch.py` — HuTuDetector marker list, detection logic
- `src/models/fusion.py` — reference for gating pattern (we'll use cross-attention instead)

---

## Phase 3: Training (10 days)

### 3.1 VAPTrainer — `src/training/trainer.py`
**3-stage training:**
1. **Audio-only** (10 epochs): Freeze text encoder, train acoustic + transformer + head. LR=1e-4
2. **Multimodal** (20 epochs): Unfreeze text encoder last 2 layers, add fusion. LR: acoustic=5e-5, text=2e-5, fusion=1e-4
3. **Full fine-tune** (20 epochs): Unfreeze all. LR: acoustic=1e-5, text=5e-6, others=5e-5

**Training details:**
- Frame-level CrossEntropyLoss over 256 classes (every frame, not just last)
- Optional: weighted loss emphasizing transition frames (±500ms around speaker changes)
- Gradient checkpointing on transformer layers
- Mixed precision (fp16)
- Batch size: 4-8 (20s windows × 16kHz = 320K samples per item)
- AdamW, cosine scheduler with warmup
- Early stopping on val loss

### 3.2 Loss — `src/training/losses.py`
- Frame-level CE loss (primary)
- Optional transition-weighted CE (upweight frames near speaker changes)
- Reuse FocalLoss from `src/training/losses.py` as alternative

**Existing code to reuse:**
- `src/training/trainer.py` — training loop structure, logging, checkpoint saving
- `src/training/losses.py` — FocalLoss implementation

---

## Phase 4: Evaluation (10 days)

### 4.1 Frame-level — `src/evaluation/frame_metrics.py`
- Cross-entropy on held-out frames
- Per-class accuracy for 256 VAP classes
- `p_now` = P(current speaker active in next 200ms) — derived from VAP probs

### 4.2 Event-level — `src/evaluation/event_metrics.py`
- Map 256-class → 3 events: Shift (yield), Hold, Backchannel
- Shift/Hold Balanced Accuracy
- Backchannel F1
- Predict-Shift AUC

### 4.3 Latency — `src/evaluation/latency_metrics.py`
- EOT latency: time from actual turn end to model prediction
- FPR at various latency thresholds
- MST-FPR curve (Minimum Silence Threshold vs False Positive Rate)

### 4.4 Evaluator — `src/evaluation/evaluator.py`
- Orchestrate all metrics
- Generate evaluation report (JSON + console summary)
- Bootstrap confidence intervals

**Existing code to reuse:**
- `docs/design/EVALUATION_METRICS.md` — complete metric implementations to adapt

---

## Phase 5: Training Runs & Ablations (20 days)

### 5.1 Pilot run
- 5-10 audio files, verify full pipeline end-to-end
- Check: labels look correct, loss decreases, gradients flow, memory fits

### 5.2 Full training
- Full dataset training with 3-stage schedule
- Target: 50 epochs total, ~2-3 days on single A100

### 5.3 Ablation studies (priority order)
1. Audio-only VAP vs Multimodal VAP (quantify text contribution)
2. Cross-Attention vs GMU fusion (justify architecture choice)
3. With/without HuTuDetector (validate discourse marker hypothesis)
4. Wav2Vec2-VI vs WavLM (acoustic encoder comparison)
5. Window size: 10s vs 20s vs 30s

---

## Phase 6: Streaming Inference (10 days)

### 6.1 `src/inference/streaming.py`
- Sliding window with overlap for continuous audio
- Audio-only path for low-latency (~61ms)
- Text integration with ASR buffer (~381ms)
- Event emission API

---

## Verification Plan

1. **Phase 0**: Unit test `encode_vap_labels` ↔ `decode_vap_labels` roundtrip
2. **Phase 1**: Run pipeline on 3 test files, visually inspect VA matrix vs audio
3. **Phase 2**: Forward pass with random data, verify output shapes (B, T, 256)
4. **Phase 3**: Overfit on 1 batch (loss → 0), then train on small set (loss decreases)
5. **Phase 4**: Evaluate with known synthetic data (perfect predictions → perfect metrics)
6. **Phase 5**: Compare ablation results, generate tables for paper
7. **Phase 6**: Real-time demo with microphone input

---

## Implementation Order & Dependencies

```
Phase 0 (utils, config)
    ↓
Phase 1 (data pipeline) ←── needs raw audio files
    ↓
Phase 2 (dataset + model) ←── needs Phase 0 + Phase 1 outputs
    ↓
Phase 3 (training) ←── needs Phase 2
    ↓
Phase 4 (evaluation) ←── can start in parallel with Phase 3
    ↓
Phase 5 (runs + ablations) ←── needs Phase 3 + Phase 4
    ↓
Phase 6 (streaming) ←── needs trained model from Phase 5
```

**Total estimated effort: ~60 days** (can overlap Phase 3/4)
