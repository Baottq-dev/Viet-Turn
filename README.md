# MM-VAP-VI: Multimodal Voice Activity Projection for Vietnamese

**Frame-level turn-taking prediction for Vietnamese conversations using multimodal deep learning.**

MM-VAP-VI predicts future voice activity for 2 speakers at 50fps (20ms frames) by encoding 2-second lookahead as 256-class labels (4 bins x 2 speakers = 8 bits).

## Architecture

```
Audio (16kHz) --> AcousticEncoder (Wav2Vec2-VI) --> (B, T, 256) --\
                                                                   --> CrossAttention --> CausalTransformer --> VAPHead --> (B, T, 256 classes)
Text (ASR)   --> LinguisticEncoder (PhoBERT + HuTuDetector) --> (B, 256) --/
```

**Components:**
- **AcousticEncoder:** Wav2Vec2-base-vi / WavLM, frozen CNN + fine-tune upper layers
- **LinguisticEncoder:** PhoBERT-base-v2 [CLS] + VAPHuTuDetector (Vietnamese discourse marker detection with n-gram matching, position-sensitive classification, exponential decay weighting)
- **Fusion:** Cross-Attention (default), GMU, or Bottleneck (Perceiver-style)
- **Temporal:** 4-layer Causal Transformer with ALiBi positional encoding
- **Output:** 256-class VAP projection (frame-level)

## Project Structure

```
Viet-Turn/
├── src/
│   ├── models/           # AcousticEncoder, LinguisticEncoder, Fusion, Transformer, ProjectionHead
│   ├── data/             # VAPDataset (sliding window), collate
│   ├── training/         # VAPTrainer (3-stage), VAPLoss, augmentation
│   ├── evaluation/       # Frame/Event/Latency metrics, MMVAPEvaluator
│   ├── inference/        # StreamingMMVAP (real-time inference)
│   └── utils/            # VAP label encode/decode, audio utilities
├── configs/              # config.yaml
├── scripts/              # Data pipeline (01-07)
├── tests/                # Unit tests
├── train.py              # Training entry point
├── docs/design/          # Architecture design, dataset guide, evaluation metrics
└── requirements.txt
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Data Pipeline

```bash
# Step 1: Speaker diarization
python scripts/01_diarize.py --input data/audio --output data/rttm

# Step 2: Build voice activity matrices
python scripts/02_build_va_matrix.py --rttm-dir data/rttm --audio-dir data/audio --output data/va_matrices

# Step 3: Generate 256-class VAP labels
python scripts/03_generate_labels.py --input data/va_matrices --output data/vap_labels

# Step 4: ASR transcription
python scripts/04_transcribe.py --input data/audio --output data/transcripts

# Step 5: Align text to frames
python scripts/05_align_text.py --transcripts data/transcripts --va-matrices data/va_matrices --output data/text_frames

# Step 6: Create train/val/test manifests
python scripts/06_create_manifest.py --audio-dir data/audio --va-dir data/va_matrices --label-dir data/vap_labels --text-dir data/text_frames --output data

# Step 7: Validate dataset
python scripts/07_validate_data.py --manifest data/vap_manifest_train.json
```

### 3. Train

```bash
# Full 3-stage training
python train.py --config configs/config.yaml

# Resume from checkpoint
python train.py --config configs/config.yaml --resume outputs/mm_vap/checkpoint_s2_e5.pt
```

### 4. Training Stages

| Stage | What | Freeze | Epochs |
|-------|------|--------|--------|
| 1 | Audio-only | Text + Fusion | 10 |
| 2 | Multimodal | Unfreeze PhoBERT top-2 + Fusion | 20 |
| 3 | Full fine-tune | All (except CNN) | 20 |

## Evaluation

4-tier evaluation framework:

| Tier | Metrics |
|------|---------|
| Frame-level | CE, Perplexity, Top-1/5 Acc, Weighted F1, ECE, Brier |
| Event-level | Shift/Hold BA, BC F1, Predict-Shift AUC |
| Latency | EOT latency, FPR, MST-FPR curve |
| Application | VAQI score |

## Key Innovation: VAPHuTuDetector

Vietnamese discourse marker detector grounded in SFP-Prosody Complementarity (Wakefield 2016):
- 5 categories: yield, hold, backchannel, turn_request, none
- N-gram matching (unigrams + bigrams + trigrams)
- Position-sensitive classification (e.g., "khong" = negation mid-sentence, question tag at end)
- Exponential position decay weighting

## Technology Stack

| Component | Choice |
|-----------|--------|
| Acoustic | Wav2Vec2-base-vi / WavLM |
| Linguistic | PhoBERT-base-v2 + HuTuDetector |
| ASR | faster-whisper (large-v3) |
| Diarization | pyannote/speaker-diarization-3.1 |
| Fusion | Cross-Attention / GMU / Bottleneck |
| Positional | ALiBi (no learned positions) |
| Loss | CE + Transition Weighting + Focal |

## License

MIT License
