# Multimodal Voice Activity Projection for Vietnamese Turn-Taking (MM-VAP-VI)

## Architecture Design Document

---

## 1. Overview

### 1.1 Research Problem

Trong hội thoại người–AI, hệ thống cần dự đoán **real-time** khi nào người nói sắp kết thúc lượt (yield), tạm dừng rồi nói tiếp (hold), hoặc đang phản hồi ngắn (backchannel) — để AI phản ứng với độ trễ tối thiểu.

### 1.2 Research Contributions

1. **Hệ thống turn-taking prediction đầu tiên cho tiếng Việt** — không có prior work nào
2. **Multimodal VAP**: Mở rộng VAP (audio-only) với linguistic branch (PhoBERT + Vietnamese discourse markers)
3. **Phân tích ảnh hưởng của thanh điệu**: Tiếng Việt có 6 thanh — F0 vừa mang nghĩa từ vựng vừa mang tín hiệu turn-taking. Self-supervised speech models có thể disentangle hai nguồn thông tin này
4. **Vietnamese discourse marker detection (Hư từ)**: Explicit modeling các hư từ tiếng Việt như tín hiệu turn-taking

### 1.3 Positioning vs. Existing Work

```
VAP (2022)           : Audio-only, self-supervised, English
Multilingual VAP (2024): Audio-only, EN/ZH/JA
Lla-VAP (2024)       : VAP + Llama LLM fusion, English
Wang et al. (2024)   : HuBERT + GPT-2, English
────────────────────────────────────────────────────
MM-VAP-VI (ours)     : Wav2Vec2/WavLM + PhoBERT + Hư từ, Vietnamese
```

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT: Stereo/Diarized Audio                     │
│                        (2-channel hoặc mono + diarization)              │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
    │  Channel 1   │  │  Channel 2   │  │  ASR (Streaming)  │
    │  (Speaker 1) │  │  (Speaker 2) │  │  PhoWhisper       │
    └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
           │                 │                   │
           ▼                 ▼                   ▼
    ┌─────────────────────────────┐    ┌──────────────────────┐
    │     ACOUSTIC ENCODER        │    │   LINGUISTIC ENCODER  │
    │  Shared Wav2Vec2/WavLM      │    │  PhoBERT + HuTuDetect │
    │                             │    │                        │
    │  CH1 → (B, T, D_a)         │    │  text → (B, D_l)      │
    │  CH2 → (B, T, D_a)         │    │  updated every 0.5s   │
    │  Concat → (B, T, 2*D_a)    │    │                        │
    └─────────────┬───────────────┘    └───────────┬────────────┘
                  │                                │
                  ▼                                ▼
    ┌──────────────────────────────────────────────────────────┐
    │              CROSS-MODAL FUSION MODULE                    │
    │                                                          │
    │  Option A: Gated Multimodal Unit (GMU)                   │
    │  Option B: Cross-Attention Transformer                   │
    │  Option C: Bottleneck Fusion                             │
    │                                                          │
    │  → (B, T, D_fused)                                       │
    └──────────────────────────┬───────────────────────────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │              PROJECTION HEAD                              │
    │                                                          │
    │  Transformer Decoder (causal, 4 layers)                  │
    │  + Linear → (B, T, 256)                                  │
    │                                                          │
    │  256 classes = 2^(2 speakers × 4 time bins)              │
    └──────────────────────────────────────────────────────────┘
```

### 2.2 Component Details

---

#### 2.2.1 Acoustic Encoder

**Primary: Pre-trained Self-Supervised Speech Model**

```
┌─────────────────────────────────────────────────────┐
│  ACOUSTIC ENCODER                                    │
│                                                      │
│  Input: raw waveform per channel (B, 1, samples)     │
│                                                      │
│  ┌───────────────────────────────────────────┐       │
│  │  Option A: wav2vec2-base-vi               │       │
│  │    - Vietnamese pre-trained               │       │
│  │    - 768-dim, 50fps (20ms/frame)          │       │
│  │    - 95M params (freeze lower layers)     │       │
│  │                                           │       │
│  │  Option B: WavLM-base                     │       │
│  │    - Multilingual, strong on paralinguistic│       │
│  │    - 768-dim, 50fps                       │       │
│  │    - 94.7M params                         │       │
│  │                                           │       │
│  │  Option C: Whisper encoder (small/medium) │       │
│  │    - Multilingual, Vietnamese included    │       │
│  │    - Already used in pipeline             │       │
│  └───────────────────────────────────────────┘       │
│                                                      │
│  Post-processing:                                    │
│    (B, T, 768) per channel                           │
│    → Concat channels: (B, T, 1536)                   │
│    → Linear projection: (B, T, 256)                  │
│    → LayerNorm                                       │
│                                                      │
│  Output: (B, T, 256) at 50fps                        │
└─────────────────────────────────────────────────────┘
```

**Tại sao Self-Supervised model thay vì hand-crafted features (Mel+F0+Energy)?**

| Tiêu chí | Hand-crafted (hiện tại) | Self-supervised (đề xuất) |
|-----------|------------------------|--------------------------|
| Thông tin thanh điệu | F0 raw — lẫn tone + intonation | Learned disentanglement qua pre-training |
| Thông tin phonetic | Không có | Có (từ pre-training trên speech) |
| Thông tin prosodic | Mel + Energy — hạn chế | Rich prosodic encoding |
| Dimension | 42 | 768 (richer representation) |
| Transferability | Không | Có (từ large-scale Vietnamese speech) |
| Tính mới cho paper | Không | Có (first Vietnamese SSL for turn-taking) |

**Ablation study nên có:**
- Wav2Vec2-VI vs WavLM vs Whisper encoder
- Frozen vs fine-tuned vs partial fine-tune (freeze bottom N layers)
- So sánh với hand-crafted features (Mel+F0+Energy + TCN) như baseline

---

#### 2.2.2 Linguistic Encoder

```
┌─────────────────────────────────────────────────────────────┐
│  LINGUISTIC ENCODER                                          │
│                                                              │
│  Input: partial transcript từ streaming ASR                  │
│         updated mỗi ~500ms                                   │
│                                                              │
│  ┌────────────────────────────────────────────────┐          │
│  │  PhoBERT-base-v2 (vinai/phobert-base-v2)      │          │
│  │  - Vietnamese BERT, 135M params                │          │
│  │  - Freeze embeddings + bottom 6 layers         │          │
│  │  - Fine-tune top 6 layers                      │          │
│  │                                                │          │
│  │  Input:  (B, seq_len) token IDs                │          │
│  │  Output: (B, seq_len, 768) token embeddings    │          │
│  │                                                │          │
│  │  Pooling strategies (ablation):                │          │
│  │    a) [CLS] token: (B, 768)                    │          │
│  │    b) Mean pooling: (B, 768)                   │          │
│  │    c) Last-K tokens: (B, K, 768)               │          │
│  │       → captures recent words = turn-final cues│          │
│  └────────────────────────────────────────────────┘          │
│                                                              │
│  ┌────────────────────────────────────────────────┐          │
│  │  HuTuDetector (Vietnamese Discourse Markers)   │          │
│  │                                                │          │
│  │  Yield markers:                                │          │
│  │    nhé, nhỉ, à, hả, ạ, đi, nha, hen,         │          │
│  │    thôi, vậy, rồi, xong, hết                  │          │
│  │                                                │          │
│  │  Hold markers:                                 │          │
│  │    mà, thì, là, nhưng, nên, vì, do,           │          │
│  │    tức là, nghĩa là, có nghĩa là              │          │
│  │                                                │          │
│  │  Backchannel markers:                          │          │
│  │    ừ, ờ, ừm, vâng, dạ, uh-huh, à há,          │          │
│  │    đúng rồi, phải, ok, được                    │          │
│  │                                                │          │
│  │  Output: (B, marker_embed_dim=64)              │          │
│  │                                                │          │
│  │  Encoding: Positional-aware                    │          │
│  │    - Marker ở cuối câu → weight cao hơn        │          │
│  │    - Marker ở đầu/giữa câu → weight thấp hơn  │          │
│  └────────────────────────────────────────────────┘          │
│                                                              │
│  Fusion:                                                     │
│    Concat [PhoBERT_output; HuTu_output]                      │
│    → Linear(768 + 64, 256)                                   │
│    → LayerNorm + GELU                                        │
│                                                              │
│  Output: (B, D_l=256)                                        │
│  → Broadcast to time: (B, T, 256)                            │
│    (giữ nguyên giữa 2 lần ASR update)                        │
└─────────────────────────────────────────────────────────────┘
```

**Streaming linguistic update protocol:**

```
Timeline (seconds):
0.0  0.5  1.0  1.5  2.0  2.5  3.0
 │    │    │    │    │    │    │
 │    ▼    │    ▼    │    ▼    │
 │  ASR₁   │  ASR₂   │  ASR₃   │
 │  "Tôi"  │  "Tôi   │  "Tôi   │
 │         │  muốn"  │  muốn   │
 │         │         │  hỏi    │
 │         │         │  nhé"   │
 │         │         │    ↑    │
 │         │         │  yield  │
 │         │         │  marker │

Audio frames: ════════════════════════════
Linguistic:   [L₁][L₁][L₂][L₂][L₃][L₃]
              (broadcast until next update)
```

---

#### 2.2.3 Cross-Modal Fusion

**3 fusion strategies (ablation study):**

```
┌─────────────────────────────────────────────────────────────┐
│  FUSION OPTIONS                                              │
│                                                              │
│  h_a: acoustic features  (B, T, 256)                        │
│  h_l: linguistic features (B, T, 256) [broadcast]           │
│                                                              │
│  ═══════════════════════════════════════════════════════     │
│  Option A: Gated Multimodal Unit (GMU) [baseline]            │
│                                                              │
│    z = σ(W_z · [h_a; h_l] + b_z)          gate ∈ [0,1]     │
│    h_fused = z ⊙ tanh(W_a · h_a) + (1-z) ⊙ tanh(W_l · h_l)│
│                                                              │
│    Output: (B, T, 256)                                       │
│    Params: ~260K                                             │
│    Pro: Lightweight, interpretable gate values                │
│    Con: No cross-modal interaction beyond gating              │
│                                                              │
│  ═══════════════════════════════════════════════════════     │
│  Option B: Cross-Attention Fusion [recommended]              │
│                                                              │
│    h_a' = MultiHeadAttn(Q=h_a, K=h_l, V=h_l) + h_a         │
│    h_l' = MultiHeadAttn(Q=h_l, K=h_a, V=h_a) + h_l         │
│    h_fused = LayerNorm(FFN([h_a'; h_l']))                    │
│                                                              │
│    Output: (B, T, 256)                                       │
│    Params: ~1.5M                                             │
│    Pro: Rich cross-modal interaction                         │
│    Con: Heavier compute                                      │
│                                                              │
│    NOTE: Causal mask trên time dimension                     │
│    để đảm bảo frame t chỉ attend đến frames ≤ t             │
│                                                              │
│  ═══════════════════════════════════════════════════════     │
│  Option C: Bottleneck Fusion (Perceiver-style)               │
│                                                              │
│    latents = LearnableTokens(N=16, dim=256)                  │
│    latents = CrossAttn(Q=latents, K=[h_a;h_l], V=[h_a;h_l]) │
│    h_fused = CrossAttn(Q=h_a, K=latents, V=latents)         │
│                                                              │
│    Output: (B, T, 256)                                       │
│    Params: ~800K                                             │
│    Pro: Scalable, fixed bottleneck                           │
│    Con: Information bottleneck có thể mất detail             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

#### 2.2.4 Temporal Context Module (Projection Head)

```
┌─────────────────────────────────────────────────────────────┐
│  TEMPORAL CONTEXT + PROJECTION HEAD                          │
│                                                              │
│  Input: h_fused (B, T, 256)                                  │
│                                                              │
│  ┌────────────────────────────────────────────────┐          │
│  │  Causal Transformer Decoder                    │          │
│  │                                                │          │
│  │  - 4 layers                                    │          │
│  │  - 8 attention heads                           │          │
│  │  - dim = 256                                   │          │
│  │  - FFN dim = 1024                              │          │
│  │  - Causal attention mask (no future)           │          │
│  │  - ALiBi positional encoding                   │          │
│  │    (no learned pos embeddings needed)          │          │
│  │  - Dropout = 0.1                               │          │
│  │                                                │          │
│  │  (B, T, 256) → (B, T, 256)                    │          │
│  └────────────────────────────────────────────────┘          │
│                                                              │
│  ┌────────────────────────────────────────────────┐          │
│  │  VAP Projection Layer                          │          │
│  │                                                │          │
│  │  Linear(256, 256) → GELU → Dropout(0.1)       │          │
│  │  → Linear(256, 256)                            │          │
│  │                                                │          │
│  │  Output: (B, T, 256)                           │          │
│  │  256 = 2^8 classes                             │          │
│  │       = 2^(2 speakers × 4 bins)               │          │
│  └────────────────────────────────────────────────┘          │
│                                                              │
│  Output: logits (B, T, 256)                                  │
│          → CrossEntropyLoss per frame                        │
└─────────────────────────────────────────────────────────────┘
```

**ALiBi (Attention with Linear Biases) thay vì sinusoidal/learned position:**
- Không cần position embedding → tiết kiệm params
- Extrapolate được đến sequence dài hơn training
- Đã được VAP gốc dùng thành công

---

#### 2.2.5 Complete Model Summary

```
┌────────────────────────────────────────────────────────────┐
│  MM-VAP-VI: Complete Model                                  │
│                                                             │
│  Acoustic Encoder:                                          │
│    Wav2Vec2-VI (frozen/partial fine-tune)      ~95M params  │
│    + Channel projection                         ~790K       │
│                                                             │
│  Linguistic Encoder:                                        │
│    PhoBERT-base-v2 (partial fine-tune)         ~135M params │
│    + HuTuDetector                                ~50K       │
│    + Projection MLP                              ~210K      │
│                                                             │
│  Cross-Modal Fusion:                                        │
│    Cross-Attention (Option B)                   ~1.5M       │
│                                                             │
│  Temporal Context:                                          │
│    Causal Transformer (4 layers)                ~4.2M       │
│                                                             │
│  Projection Head:                                           │
│    MLP → 256 classes                             ~130K      │
│                                                             │
│  ─────────────────────────────────────────────────          │
│  Total (trainable, with frozen SSL + BERT):     ~6.9M      │
│  Total (all params):                            ~237M       │
│                                                             │
│  Inference speed target:                                    │
│    > 50fps (real-time at 20ms/frame) on GPU                 │
│    > 20fps on CPU (acceptable for streaming)                │
└────────────────────────────────────────────────────────────┘
```

---

## 3. VAP Label Generation

### 3.1 Label Construction Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  LABEL GENERATION PIPELINE (Self-supervised)                 │
│                                                              │
│  Input: Audio file + Speaker diarization                     │
│                                                              │
│  Step 1: Voice Activity Detection per speaker                │
│  ┌─────────────────────────────────────────────┐             │
│  │  pyannote/audio pipeline hoặc WhisperX      │             │
│  │  → segments: [(spk, start, end), ...]       │             │
│  │  → frame-level binary matrix:               │             │
│  │     VA[speaker][frame] ∈ {0, 1}             │             │
│  │     at 50fps (20ms per frame)               │             │
│  └─────────────────────────────────────────────┘             │
│                                                              │
│  Step 2: Future projection at each frame                     │
│  ┌─────────────────────────────────────────────┐             │
│  │  Projection window: 2 seconds (100 frames)  │             │
│  │  Divided into 4 bins:                        │             │
│  │                                              │             │
│  │  Bin 0: [t, t+200ms)      → 10 frames       │             │
│  │  Bin 1: [t+200ms, t+600ms)  → 20 frames     │             │
│  │  Bin 2: [t+600ms, t+1200ms) → 30 frames     │             │
│  │  Bin 3: [t+1200ms, t+2000ms) → 40 frames    │             │
│  │                                              │             │
│  │  For each bin b, for each speaker s:         │             │
│  │    ratio = mean(VA[s][bin_start:bin_end])     │             │
│  │    active[s][b] = 1 if ratio > 0.5 else 0    │             │
│  │                                              │             │
│  │  label_vector = [                            │             │
│  │    sp1_bin0, sp1_bin1, sp1_bin2, sp1_bin3,   │             │
│  │    sp2_bin0, sp2_bin1, sp2_bin2, sp2_bin3    │             │
│  │  ]  → 8 bits → index ∈ [0, 255]             │             │
│  └─────────────────────────────────────────────┘             │
│                                                              │
│  Step 3: Map to class index                                  │
│  ┌─────────────────────────────────────────────┐             │
│  │  class_index = Σ(bit_i × 2^i) for i in 0..7 │             │
│  │                                              │             │
│  │  Ví dụ: SP1 nói xong, SP2 sắp nói           │             │
│  │  sp1: [1, 1, 0, 0]  sp2: [0, 0, 1, 1]       │             │
│  │  bits: [1,1,0,0, 0,0,1,1]                   │             │
│  │  index: 1+2+0+0 +0+0+64+128 = 195           │             │
│  │                                              │             │
│  │  Ví dụ: SP1 hold (tiếp tục nói)             │             │
│  │  sp1: [1, 1, 1, 1]  sp2: [0, 0, 0, 0]       │             │
│  │  bits: [1,1,1,1, 0,0,0,0]                   │             │
│  │  index: 1+2+4+8 = 15                        │             │
│  │                                              │             │
│  │  Ví dụ: Backchannel (SP2 nói ngắn)          │             │
│  │  sp1: [1, 1, 1, 1]  sp2: [0, 1, 0, 0]       │             │
│  │  bits: [1,1,1,1, 0,1,0,0]                   │             │
│  │  index: 1+2+4+8 +0+32+0+0 = 47              │             │
│  └─────────────────────────────────────────────┘             │
│                                                              │
│  Output: labels tensor (num_frames,) dtype=long              │
│          mỗi giá trị ∈ [0, 255]                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Label Distribution Analysis

Trong hội thoại tự nhiên, phân bố các class rất không đều:

```
Phổ biến nhất:
  - SP1 nói liên tục, SP2 im    : class 15  (SP1=[1,1,1,1], SP2=[0,0,0,0])
  - SP2 nói liên tục, SP1 im    : class 240 (SP1=[0,0,0,0], SP2=[1,1,1,1])
  - Cả hai im lặng              : class 0   (all zeros)

Hiếm gặp nhưng quan trọng:
  - Turn transitions (shift)     : ~5-10% frames
  - Backchannel overlaps         : ~2-5% frames
  - Competitive overlaps         : ~1-3% frames
```

**Xử lý class imbalance:**
- Focal Loss (γ=2.0) — tập trung vào hard examples
- Hoặc class weighting dựa trên inverse frequency
- Hoặc nhóm 256 classes thành clusters rồi hierarchical classification

### 3.3 Mapping VAP Classes → Turn-Taking Events (Evaluation)

```python
# Ánh xạ từ 256-class prediction → turn-taking events
def vap_to_events(probs, threshold=0.5):
    """
    probs: (T, 256) — softmax probabilities at each frame

    Returns per-frame predictions for:
      - P(shift):       SP1 sẽ dừng, SP2 sẽ nói
      - P(hold):        SP1 tiếp tục, SP2 im
      - P(backchannel): SP2 nói ngắn, SP1 vẫn nói
      - P(overlap):     cả hai cùng nói
    """
    # Group classes by event type
    # shift: SP1 bins [1,1,0,0] or [1,0,0,0], SP2 bins [0,0,1,1] or [0,1,1,1]
    # hold:  SP1 bins [1,1,1,1], SP2 bins [0,0,0,0]
    # backchannel: SP1 bins [1,1,1,1], SP2 bins [0,1,0,0] or [1,0,0,0]

    shift_classes = get_shift_class_indices()
    hold_classes = get_hold_class_indices()
    bc_classes = get_backchannel_class_indices()

    p_shift = probs[:, shift_classes].sum(dim=-1)
    p_hold = probs[:, hold_classes].sum(dim=-1)
    p_bc = probs[:, bc_classes].sum(dim=-1)

    return p_shift, p_hold, p_bc
```

---

## 4. Data Pipeline

### 4.1 Data Collection Requirements

```
┌─────────────────────────────────────────────────────────────┐
│  DATA REQUIREMENTS                                           │
│                                                              │
│  Audio specifications:                                       │
│    - Format: WAV, 16kHz, mono hoặc stereo                    │
│    - Nếu mono: cần speaker diarization pipeline              │
│    - Nếu stereo: mỗi channel 1 speaker (ideal)              │
│    - Minimum quality: 16kHz, minimal background noise        │
│                                                              │
│  Content types (by priority):                                │
│    1. Podcast tiếng Việt (2 người, đối thoại tự nhiên)       │
│    2. Phỏng vấn (interviewer + guest)                        │
│    3. Talk show / debate                                     │
│    4. Gọi điện thoại (nếu có corpus)                         │
│                                                              │
│  Scale targets:                                              │
│    Phase 1 (proof of concept): 20-30 giờ                     │
│    Phase 2 (full training):    50-100 giờ                    │
│    Phase 3 (scaling):          200+ giờ                      │
│                                                              │
│  Diversity requirements:                                     │
│    - Cân bằng giới tính (nam-nam, nữ-nữ, nam-nữ)            │
│    - Đa dạng chủ đề                                          │
│    - Bao gồm phương ngữ nếu có thể (Bắc, Trung, Nam)       │
│    - Đa dạng phong cách (formal vs casual)                   │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Data Processing Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  FULL DATA PIPELINE                                           │
│                                                               │
│  Raw Audio (YouTube/Podcast)                                  │
│       │                                                       │
│       ▼                                                       │
│  ┌──────────────────┐                                         │
│  │ 1. Audio Cleanup │                                         │
│  │   - Noise reduction (noisereduce / DeepFilterNet)          │
│  │   - Normalize volume                                       │
│  │   - Resample to 16kHz mono                                 │
│  └────────┬─────────┘                                         │
│           │                                                   │
│           ▼                                                   │
│  ┌──────────────────────────────────┐                         │
│  │ 2. Speaker Diarization           │                         │
│  │   Primary: pyannote/speaker-diarization-3.1                │
│  │   Backup:  WhisperX diarization                            │
│  │                                                            │
│  │   Output: segments with speaker IDs                        │
│  │   [(spk_id, start_sec, end_sec), ...]                      │
│  └────────┬─────────────────────────┘                         │
│           │                                                   │
│           ▼                                                   │
│  ┌──────────────────────────────────┐                         │
│  │ 3. Voice Activity Matrix          │                         │
│  │                                    │                        │
│  │   frame_rate = 50  # fps (20ms)    │                        │
│  │   total_frames = duration × 50     │                        │
│  │                                    │                        │
│  │   va_matrix[speaker][frame] = 1    │                        │
│  │   if speaker is active at frame    │                        │
│  │                                    │                        │
│  │   Shape: (num_speakers, total_frames)                      │
│  └────────┬─────────────────────────┘                         │
│           │                                                   │
│           ▼                                                   │
│  ┌──────────────────────────────────┐                         │
│  │ 4. VAP Label Generation           │                         │
│  │                                    │                        │
│  │   For each frame t:                │                        │
│  │     Look ahead 2 seconds           │                        │
│  │     Compute 4-bin activity          │                        │
│  │     Encode as class index [0-255]   │                        │
│  │                                    │                        │
│  │   Output: labels (total_frames,)    │                        │
│  └────────┬─────────────────────────┘                         │
│           │                                                   │
│           ▼                                                   │
│  ┌──────────────────────────────────┐                         │
│  │ 5. ASR Transcription              │                         │
│  │                                    │                        │
│  │   PhoWhisper → word-level timestamps                       │
│  │   Per segment: {text, words: [{word, start, end}]}         │
│  │                                    │                        │
│  │   Align words with audio frames    │                        │
│  │   → text_at_frame[t] = transcript up to frame t            │
│  └────────┬─────────────────────────┘                         │
│           │                                                   │
│           ▼                                                   │
│  ┌──────────────────────────────────┐                         │
│  │ 6. Dataset Storage                 │                        │
│  │                                    │                        │
│  │   Per audio file, save:            │                        │
│  │     audio.wav          (raw audio) │                        │
│  │     va_matrix.npy      (2, T)      │                        │
│  │     vap_labels.npy     (T,)        │                        │
│  │     transcript.json    (word-level timestamps)             │
│  │     metadata.json      (duration, speakers, source)        │
│  └──────────────────────────────────┘                         │
│                                                               │
│  Train / Val / Test split:                                    │
│    - Split by conversation (không split trong 1 conversation) │
│    - 80% / 10% / 10%                                          │
│    - Stratify theo conversation type + speaker gender          │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 Dataset Class (New Design)

```python
class VAPDataset(torch.utils.data.Dataset):
    """
    Produces overlapping windows of audio + VAP labels.
    """
    def __init__(
        self,
        data_dir: str,            # directory with processed files
        window_sec: float = 20.0, # context window in seconds
        stride_sec: float = 5.0,  # stride between windows
        frame_rate: int = 50,     # frames per second (20ms)
        sample_rate: int = 16000,
        include_text: bool = True,
        asr_update_interval: float = 0.5  # seconds
    ):
        ...

    def __getitem__(self, idx):
        """
        Returns:
            audio_ch1: (samples,)     — speaker 1 audio (or mono)
            audio_ch2: (samples,)     — speaker 2 audio (or zeros if mono)
            vap_labels: (T,)          — per-frame VAP class index [0-255]
            text_features: {
                'input_ids': (num_updates, seq_len),
                'attention_mask': (num_updates, seq_len),
                'update_frames': (num_updates,)  — at which frame each text update occurs
            }
            metadata: dict
        """
        ...

    def collate_fn(self, batch):
        """
        Pads to max window length in batch.
        Returns:
            audio_ch1: (B, max_samples)
            audio_ch2: (B, max_samples)
            vap_labels: (B, max_T)
            text_features: batched dict
            padding_mask: (B, max_T)   — True for valid frames
        """
        ...
```

---

## 5. Training Procedure

### 5.1 Training Configuration

```yaml
# training_config_vap.yaml

model:
  acoustic_encoder: "wav2vec2-base-vi"  # hoặc "wavlm-base"
  acoustic_freeze_layers: 8              # freeze bottom 8/12 layers
  acoustic_output_dim: 256

  linguistic_encoder: "vinai/phobert-base-v2"
  linguistic_freeze_layers: 6            # freeze bottom 6/12 layers
  linguistic_output_dim: 256
  marker_embedding_dim: 64

  fusion_type: "cross_attention"         # gmu | cross_attention | bottleneck
  fusion_dim: 256
  fusion_heads: 8

  transformer_layers: 4
  transformer_heads: 8
  transformer_dim: 256
  transformer_ffn_dim: 1024
  transformer_dropout: 0.1

  num_vap_classes: 256
  projection_bins: [200, 400, 600, 800]  # ms, total = 2000ms

training:
  epochs: 50
  batch_size: 16                         # smaller due to larger model
  gradient_accumulation_steps: 4         # effective batch = 64
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0

  scheduler: cosine_with_warmup

  # Different LR for different components
  lr_acoustic_encoder: 1e-5              # fine-tune slowly
  lr_linguistic_encoder: 2e-5            # fine-tune slowly
  lr_new_modules: 1e-4                   # fusion + projection train faster

loss:
  type: cross_entropy                    # per-frame CE over 256 classes
  label_smoothing: 0.1
  # Hoặc:
  # type: focal
  # gamma: 2.0

data:
  window_sec: 20.0
  stride_sec: 5.0
  frame_rate: 50
  sample_rate: 16000
  asr_update_interval: 0.5

augmentation:
  # Audio augmentation
  noise_injection:
    enabled: true
    snr_range: [10, 30]
  time_stretch:
    enabled: true
    range: [0.9, 1.1]

  # Modality dropout (novel regularization)
  text_dropout: 0.3                      # 30% chance bỏ text → audio-only
  audio_channel_dropout: 0.1             # 10% chance zero 1 channel

  # SpecAugment-style
  time_masking:
    enabled: true
    max_frames: 20

evaluation:
  # VAP zero-shot event evaluation
  shift_threshold: 0.5
  hold_threshold: 0.5
  backchannel_threshold: 0.3

  # Latency evaluation
  latency_window_ms: 1000               # measure within 1s of actual event
```

### 5.2 Training Loop (Pseudocode)

```python
class VAPTrainer:
    def train_epoch(self):
        for batch in train_loader:
            audio_ch1, audio_ch2, vap_labels, text_features, mask = batch

            # 1. Acoustic encoding (both channels)
            h_a1 = self.model.acoustic_encoder(audio_ch1)  # (B, T, 256)
            h_a2 = self.model.acoustic_encoder(audio_ch2)  # (B, T, 256)
            h_acoustic = torch.cat([h_a1, h_a2], dim=-1)   # (B, T, 512)
            h_acoustic = self.model.acoustic_proj(h_acoustic)  # (B, T, 256)

            # 2. Linguistic encoding (periodic updates)
            h_linguistic = self.model.encode_text_streaming(
                text_features  # includes update_frames for alignment
            )  # (B, T, 256) — broadcast between updates

            # 3. Fusion
            h_fused = self.model.fusion(h_acoustic, h_linguistic)  # (B, T, 256)

            # 4. Temporal context + projection
            logits = self.model.projection_head(h_fused)  # (B, T, 256)

            # 5. Loss (per-frame, masked)
            # logits: (B, T, 256), labels: (B, T)
            logits_flat = logits[mask]          # (N_valid, 256)
            labels_flat = vap_labels[mask]      # (N_valid,)
            loss = F.cross_entropy(logits_flat, labels_flat,
                                   label_smoothing=0.1)

            # 6. Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
```

### 5.3 Multi-Stage Training Strategy

```
Stage 1: Audio-only pre-training (5-10 epochs)
  - Freeze linguistic encoder entirely
  - Train acoustic encoder (top layers) + fusion + projection
  - Mục đích: model học cơ bản về turn-taking từ prosody

Stage 2: Multimodal fine-tuning (20-30 epochs)
  - Unfreeze linguistic encoder (top layers)
  - Joint training tất cả components
  - Giảm LR cho acoustic encoder

Stage 3: Full fine-tuning (10-15 epochs)
  - Unfreeze thêm layers nếu data đủ lớn
  - Giảm LR tổng thể
  - Early stopping dựa trên val event-based metrics
```

---

## 6. Evaluation Protocol

### 6.1 Intrinsic Evaluation (Per-frame)

```
┌─────────────────────────────────────────────────────────────┐
│  FRAME-LEVEL METRICS                                         │
│                                                              │
│  1. Cross-Entropy Loss (val set)                             │
│     — Đo chất lượng prediction phân bố 256 classes           │
│                                                              │
│  2. Top-1 Accuracy                                           │
│     — % frames dự đoán đúng class index                      │
│     — Kỳ vọng: thấp (256 classes, long-tail distribution)    │
│                                                              │
│  3. Weighted F1                                              │
│     — F1 trung bình có trọng số theo class frequency         │
│                                                              │
│  4. Perplexity                                               │
│     — exp(CE loss) — dễ so sánh giữa các model              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Event-Based Evaluation (Zero-shot, theo chuẩn VAP)

```
┌─────────────────────────────────────────────────────────────┐
│  EVENT-BASED METRICS (Primary metrics cho paper)             │
│                                                              │
│  Cách đánh giá: Tại mỗi sự kiện turn-taking trong test set, │
│  đo model dự đoán đúng hay sai.                             │
│                                                              │
│  ═══════════════════════════════════════════════════════     │
│  Event 1: SHIFT (turn transition)                            │
│  ─────────────────────────────────                           │
│  Definition: SP1 dừng nói, SP2 bắt đầu nói                  │
│  Evaluation window: silence onset → 200ms after SP2 starts   │
│                                                              │
│  Metric: Balanced Accuracy                                   │
│    - Model dự đoán "SP2 sẽ active" đúng không?              │
│    - Compare P(next_speaker=SP2) vs P(next_speaker=SP1)      │
│                                                              │
│  ═══════════════════════════════════════════════════════     │
│  Event 2: HOLD (within-turn pause)                           │
│  ─────────────────────────────────                           │
│  Definition: SP1 tạm dừng ≥ 200ms rồi nói tiếp              │
│  Evaluation window: pause onset → SP1 resumes                │
│                                                              │
│  Metric: Balanced Accuracy                                   │
│    - Model dự đoán "SP1 sẽ tiếp tục" đúng không?            │
│                                                              │
│  ═══════════════════════════════════════════════════════     │
│  Event 3: BACKCHANNEL                                        │
│  ─────────────────────────────────                           │
│  Definition: SP2 nói ngắn (< 1s) trong khi SP1 giữ lượt     │
│  Evaluation window: 1s before BC onset                       │
│                                                              │
│  Metric: F1-score                                            │
│    - Model dự đoán "SP2 sẽ nói ngắn" đúng không?            │
│    - Precision: tránh false alarm                            │
│    - Recall: phát hiện đủ backchannels                       │
│                                                              │
│  ═══════════════════════════════════════════════════════     │
│  Event 4: OVERLAP (optional, advanced)                       │
│  ─────────────────────────────────                           │
│  Definition: Cả hai speakers nói cùng lúc > 200ms            │
│  Metric: Onset prediction accuracy                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Latency Evaluation (Application-oriented)

```
┌─────────────────────────────────────────────────────────────┐
│  LATENCY METRICS                                             │
│                                                              │
│  Scenario: Human-AI conversation                             │
│  Goal: AI phản ứng nhanh nhất có thể khi human hết lượt     │
│                                                              │
│  1. End-of-Turn Detection Latency                            │
│     ─────────────────────────────                            │
│     Measure: Time from actual turn-end to model detecting    │
│              shift with confidence > threshold               │
│                                                              │
│     P(shift) > θ at frame t                                  │
│     actual_turn_end at frame t*                              │
│     latency = (t - t*) × 20ms                               │
│                                                              │
│     Report: median, mean, P90, P95                           │
│     Target: median < 200ms (human-like response time)        │
│                                                              │
│  2. Early Detection Rate                                     │
│     ───────────────────────                                  │
│     % of turn-ends detected BEFORE actual silence            │
│     (model predicts shift while speaker is still speaking    │
│      final syllable → anticipatory, like humans)             │
│                                                              │
│  3. False Endpoint Rate (FPR)                                │
│     ──────────────────────────                               │
│     % of hold pauses incorrectly classified as shifts        │
│     Report at multiple thresholds: θ ∈ {0.3, 0.5, 0.7}      │
│                                                              │
│  4. Latency vs FPR Curve                                     │
│     ──────────────────────────                               │
│     Vary θ from 0.1 to 0.9                                   │
│     Plot: x=FPR, y=median_latency                            │
│     Area Under Curve → single number for comparison          │
│                                                              │
│  5. Mean Shift Time (MST) @ fixed FPR                        │
│     ────────────────────────────────────                     │
│     At FPR = 0.05 (5%), what is the mean detection latency?  │
│     Cho phép so sánh công bằng giữa các model                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.4 Ablation Study Design

```
┌─────────────────────────────────────────────────────────────┐
│  ABLATION EXPERIMENTS                                        │
│                                                              │
│  A1: Acoustic encoder comparison                             │
│      a) Wav2Vec2-VI (Vietnamese pre-trained)                 │
│      b) WavLM-base (multilingual)                            │
│      c) Whisper encoder (small)                              │
│      d) Hand-crafted: Mel+F0+Energy + TCN (baseline hiện tại)│
│                                                              │
│  A2: Modality ablation                                       │
│      a) Audio-only (no text) — so sánh trực tiếp với VAP    │
│      b) Text-only (no audio) — PhoBERT + discourse markers  │
│      c) Audio + Text (full model)                            │
│      → Đo contribution của mỗi modality                     │
│                                                              │
│  A3: Fusion strategy comparison                              │
│      a) GMU (gated fusion)                                   │
│      b) Cross-Attention                                      │
│      c) Bottleneck (Perceiver-style)                         │
│      d) Simple concatenation (baseline)                      │
│                                                              │
│  A4: Linguistic features                                     │
│      a) PhoBERT only                                         │
│      b) HuTuDetector only                                    │
│      c) PhoBERT + HuTuDetector (full)                        │
│      → Đo giá trị của explicit discourse marker detection    │
│                                                              │
│  A5: Projection window analysis                              │
│      a) 1 second (2 bins)                                    │
│      b) 2 seconds (4 bins) — standard VAP                    │
│      c) 3 seconds (6 bins)                                   │
│      → Tìm optimal projection horizon cho tiếng Việt        │
│                                                              │
│  A6: Vietnamese-specific analysis                            │
│      a) Performance by dialect (Bắc/Trung/Nam)              │
│      b) Performance by conversation type                     │
│      c) Impact of tonal F0 on prediction accuracy            │
│      d) Discourse marker analysis: which markers are         │
│         most predictive? Position effect (đầu/giữa/cuối)?   │
│                                                              │
│  A7: Data scale analysis                                     │
│      a) 10h → 20h → 50h → 100h                              │
│      → Scaling curve, minimum viable data size               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Inference Pipeline (Streaming)

### 7.1 Real-time Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  STREAMING INFERENCE PIPELINE                                    │
│                                                                  │
│  Audio Input (microphone, 16kHz)                                 │
│       │                                                          │
│       ├──────────────────────┐                                   │
│       ▼                      ▼                                   │
│  ┌──────────┐         ┌──────────────┐                           │
│  │ Audio    │         │ ASR Engine   │                           │
│  │ Buffer   │         │ (PhoWhisper) │                           │
│  │ (20ms    │         │              │                           │
│  │  frames) │         │ Output every │                           │
│  └────┬─────┘         │ ~500ms       │                           │
│       │               └──────┬───────┘                           │
│       │                      │                                   │
│       ▼                      ▼                                   │
│  ┌──────────┐         ┌──────────────┐                           │
│  │ Acoustic │         │ Linguistic   │                           │
│  │ Encoder  │         │ Encoder      │                           │
│  │          │         │              │                           │
│  │ Cached   │         │ Re-encode    │                           │
│  │ states   │         │ on new text  │                           │
│  └────┬─────┘         └──────┬───────┘                           │
│       │                      │                                   │
│       ▼                      ▼                                   │
│  ┌─────────────────────────────────────┐                         │
│  │           Fusion + Transformer       │                        │
│  │           (cached KV states)         │                        │
│  │                                      │                        │
│  │  New frame → attend to cached past   │                        │
│  │  → output prediction for this frame  │                        │
│  └──────────────────┬──────────────────┘                         │
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────┐                         │
│  │  Decision Logic                      │                        │
│  │                                      │                        │
│  │  p_shift = sum(probs[shift_classes]) │                        │
│  │                                      │                        │
│  │  if p_shift > θ_high (e.g., 0.7):   │                        │
│  │    → AI starts responding            │                        │
│  │                                      │                        │
│  │  if p_shift > θ_low (e.g., 0.4):    │                        │
│  │    → AI prepares response (pre-gen)  │                        │
│  │                                      │                        │
│  │  if p_backchannel > 0.5:             │                        │
│  │    → AI gives backchannel ("ừm")     │                        │
│  │                                      │                        │
│  │  else:                               │                        │
│  │    → Continue listening               │                        │
│  └─────────────────────────────────────┘                         │
│                                                                  │
│  Target latency budget:                                          │
│    Audio encoding:    ~10ms                                      │
│    Fusion+Transformer: ~5ms                                      │
│    Decision logic:     ~1ms                                      │
│    ──────────────────────────                                    │
│    Total per frame:   ~16ms (< 20ms frame interval → real-time) │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 State Caching for Streaming

```python
class StreamingMMVAP:
    """Stateful streaming wrapper for inference."""

    def __init__(self, model, max_context_frames=500):
        self.model = model
        self.max_context = max_context_frames  # 10s at 50fps

        # Cached states
        self.audio_buffer = []           # raw audio frames
        self.acoustic_cache = []         # encoded acoustic features
        self.linguistic_cache = None     # latest text encoding
        self.transformer_kv_cache = None # transformer KV cache

    def process_audio_frame(self, audio_frame):
        """Process one 20ms audio frame. Returns prediction."""

        # 1. Encode audio frame (with context)
        self.audio_buffer.append(audio_frame)
        h_a = self.model.acoustic_encoder.encode_frame(
            self.audio_buffer[-self.max_context:]
        )  # (1, 1, 256) — single frame output

        # 2. Get linguistic features (cached, updated externally)
        h_l = self.linguistic_cache  # (1, 1, 256)
        if h_l is None:
            h_l = torch.zeros(1, 1, 256)

        # 3. Fuse
        h_fused = self.model.fusion(h_a, h_l)  # (1, 1, 256)

        # 4. Transformer with KV cache
        logits, self.transformer_kv_cache = self.model.transformer(
            h_fused,
            past_kv=self.transformer_kv_cache
        )  # (1, 1, 256)

        # 5. Prediction
        probs = F.softmax(logits[0, 0], dim=-1)  # (256,)
        return self.decode_vap_probs(probs)

    def update_text(self, partial_transcript: str):
        """Called when ASR produces new output (~every 500ms)."""
        self.linguistic_cache = self.model.linguistic_encoder(
            partial_transcript
        )  # (1, 1, 256)
```

---

## 8. Experiments & Baselines

### 8.1 Baselines to Compare Against

```
┌─────────────────────────────────────────────────────────────┐
│  BASELINES                                                   │
│                                                              │
│  B1: Random baseline                                         │
│      Predict most frequent class at each frame               │
│      → Lower bound                                          │
│                                                              │
│  B2: VAD-only baseline                                       │
│      Silence > 700ms → predict shift                         │
│      Silence 200-700ms → predict hold                        │
│      → Traditional endpointing                              │
│                                                              │
│  B3: VAP (audio-only, original architecture)                 │
│      CPC encoder + Transformer                               │
│      Retrained on Vietnamese data                            │
│      → Direct comparison: does multimodal help?              │
│                                                              │
│  B4: Text-only baseline                                      │
│      PhoBERT + classifier                                    │
│      Predict turn-taking from text only (given ASR output)   │
│      → How much does text alone tell us?                     │
│                                                              │
│  B5: Current Viet-Turn (turn classification)                 │
│      TCN + PhoBERT, segment-level                            │
│      Map to events post-hoc                                  │
│      → Comparison with existing approach                     │
│                                                              │
│  B6: Krisp-style (audio-only, lightweight)                   │
│      Small CNN/GRU on prosodic features                      │
│      → Latency-optimized baseline                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Expected Results Table (Template)

```
┌──────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Model            │ Shift BA │ Hold BA  │ BC F1    │ Latency  │ FPR@0.5  │
│                  │          │          │          │ (median) │          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ B1: Random       │ 50.0%    │ 50.0%    │  ~0%     │   N/A    │  50%     │
│ B2: VAD-only     │ ~65%     │ ~60%     │  ~0%     │  700ms   │  ~15%    │
│ B3: VAP (audio)  │ ~78%     │ ~72%     │ ~45%     │  ~300ms  │  ~8%     │
│ B4: Text-only    │ ~72%     │ ~68%     │ ~35%     │  ~500ms  │  ~12%    │
│ B5: Viet-Turn v1 │ ~70%     │ ~65%     │ ~40%     │   N/A*   │   N/A*   │
│ B6: Krisp-style  │ ~74%     │ ~70%     │ ~30%     │  ~400ms  │  ~10%    │
├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ MM-VAP-VI (ours) │ ~82%     │ ~76%     │ ~52%     │  ~200ms  │  ~5%     │
└──────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

* Viet-Turn v1 là segment-level → không có frame-level latency metrics
BA = Balanced Accuracy
BC = Backchannel
FPR = False Positive Rate (sai "hết lượt" khi thực ra là pause)
```

---

## 9. Implementation Roadmap

### Phase 1: Data Pipeline

```
[ ] Thu thập 20-30h audio hội thoại tiếng Việt (YouTube podcasts)
[ ] Chạy speaker diarization (pyannote 3.1)
[ ] Tạo voice activity matrix per speaker
[ ] Implement VAP label generation
[ ] Chạy PhoWhisper ASR → word-level timestamps
[ ] Implement VAPDataset class
[ ] Verify label distribution statistics
[ ] Train/val/test split (by conversation)
```

### Phase 2: Model Implementation

```
[ ] Implement acoustic encoder wrapper (Wav2Vec2/WavLM)
[ ] Adapt linguistic encoder for streaming text
[ ] Implement 3 fusion options (GMU, CrossAttn, Bottleneck)
[ ] Implement causal Transformer with ALiBi
[ ] Implement VAP projection head
[ ] Implement VAP class ↔ event mapping
[ ] Unit tests cho mỗi component
```

### Phase 3: Training

```
[ ] Implement VAPTrainer with frame-level loss
[ ] Implement multi-stage training schedule
[ ] Train audio-only baseline (VAP reproduction)
[ ] Train text-only baseline
[ ] Train full multimodal model
[ ] Hyperparameter tuning
```

### Phase 4: Evaluation

```
[ ] Implement event detection from VAP predictions
[ ] Implement all evaluation metrics
[ ] Run ablation studies (A1-A7)
[ ] Generate latency vs FPR curves
[ ] Vietnamese-specific analysis (dialect, discourse markers)
[ ] Statistical significance tests
```

### Phase 5: Streaming Demo

```
[ ] Implement StreamingMMVAP wrapper
[ ] Build real-time demo with microphone input
[ ] Measure end-to-end latency
[ ] (Optional) Integrate with Vietnamese voice AI system
```

---

## 10. Related Work & References

### 10.1 Core Turn-Taking Prediction

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 1 | Ekstedt & Skantze, "Voice Activity Projection: Self-supervised Learning of Turn-taking Events" | Interspeech | 2022 | Introduced VAP framework — self-supervised 256-class projection of future voice activity. Foundation of current SOTA. |
| 2 | Ekstedt & Skantze, "TurnGPT: a Transformer-based Language Model for Predicting Turn-taking in Spoken Dialog" | Findings of EMNLP | 2020 | Text-only turn prediction via transformer LM. Showed syntactic/pragmatic completeness predicts turns. Precursor to VAP. |
| 3 | Inoue et al., "Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection" | IWSDS | 2024 | Extended VAP for real-time streaming. Showed 1-second context sufficient for CPU real-time. arXiv:2401.04868 |
| 4 | Inoue et al., "Multilingual Turn-taking Prediction Using Voice Activity Projection" | LREC-COLING | 2024 | VAP trained on English + Mandarin + Japanese. Cross-lingual transfer works, including for tonal languages. arXiv:2403.06487 |
| 5 | Ekstedt et al., "How Much Context Does My Attention-Based ASR System Need?" | Interspeech | 2023 | Analysis of context requirements for speech models — relevant to window size selection. |

### 10.2 Multimodal Turn-Taking

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 6 | Jiang et al., "Lla-VAP: LSTM Ensemble of Llama and VAP for Turn-Taking Prediction" | arXiv | 2024 | First fusion of LLM (Llama) with VAP via LSTM ensemble. Showed linguistic features improve acoustic-only VAP. arXiv:2412.18061 |
| 7 | Wang et al., "Turn-taking and Backchannel Prediction with Acoustic and Large Language Model Fusion" | ICASSP | 2024 | HuBERT + GPT-2/RedPajama fusion. Multi-task instruction fine-tuning. AUC 0.8785 on Switchboard. |
| 8 | Onishi et al., "Multimodal Voice Activity Projection for Turn-Taking" | IEICE Transactions | 2025 | Extended VAP with visual (gaze, gesture) and linguistic modalities. |
| 9 | Roddy et al., "Multimodal Continuous Turn-Taking Prediction Using Multiscale RNNs" | ICMI | 2018 | Early multimodal approach: audio + language + visual features with hierarchical RNN. |

### 10.3 End-of-Turn Detection (Industry/Applied)

| # | Paper / Blog | Source | Year | Key Contribution |
|---|-------------|--------|------|-----------------|
| 10 | "SpeculativeETD: Speculative End-of-Turn Detection with On-Device and Server Collaboration" | arXiv:2503.23439 | 2025 | Hybrid on-device GRU (1M) + server Wav2Vec2 (94M). ETD dataset: 120k+ samples, 300+ hours. |
| 11 | "Using a Transformer to Improve End-of-Turn Detection" | LiveKit Blog | 2024 | Fine-tuned SmolLM (135M) on text from STT. Sliding 4-turn window. 39% interruption reduction. |
| 12 | "Improved End-of-Turn Model Cuts Voice AI Interruptions 39%" | LiveKit Blog | 2025 | LiveKit EOU v2: Qwen2.5-0.5B distilled from 7B teacher. ~25ms inference. |
| 13 | "Turn-Taking Model for Voice AI" | Krisp Blog | 2024 | Audio-only, 6.1M params. Prosodic features. Mean Shift Time analysis. |
| 14 | "Intelligent Turn Detection for Universal-Streaming" | AssemblyAI Blog | 2025 | Integrated <EoT> token in STT model. ~300ms total latency. |
| 15 | "Evaluating End-of-Turn Detection Models" | Deepgram Blog | 2025 | Novel sequence-based evaluation methodology. VAQI metric. |

### 10.4 Self-Supervised Speech Representations

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 16 | Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" | NeurIPS | 2020 | Foundation SSL model for speech. Contrastive learning on masked latent representations. |
| 17 | Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing" | IEEE JSTSP | 2022 | Strong on paralinguistic tasks (emotion, speaker). Denoising pre-training objective. |
| 18 | Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units" | IEEE/ACM TASLP | 2021 | Offline clustering + masked prediction. Strong on downstream tasks. |
| 19 | "Prosodic and Lexical Cues in Turn-Taking Prediction Using Self-Supervised Speech Representations" | arXiv:2601.13835 | 2026 | Analysis of prosody vs lexicon in SSL features for turn-taking. Found intensity > pitch for prediction. |

### 10.5 Vietnamese NLP & Speech

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 20 | Nguyen & Nguyen, "PhoBERT: Pre-trained Language Models for Vietnamese" | Findings of EMNLP | 2020 | Vietnamese BERT. SOTA on Vietnamese NLP tasks. |
| 21 | Nguyen et al., "PhoWhisper: Automatic Speech Recognition for Vietnamese" | arXiv | 2023 | Vietnamese fine-tuned Whisper. Strong ASR baseline for Vietnamese. |
| 22 | Trang et al., "Prosodic Boundary Prediction for Vietnamese TTS" | Interspeech | 2021 | Vietnamese prosodic boundary model. 6-10% TTS naturalness improvement. Related to turn boundary prediction. |
| 23 | Luong & Vu, "A non-expert Kaldi recipe for Vietnamese Speech Recognition System" | WNUT | 2016 | Early Vietnamese ASR system. Useful for historical context. |

### 10.6 Conversational Analysis & Turn-Taking Theory

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 24 | Sacks et al., "A Simplest Systematics for the Organization of Turn-Taking for Conversation" | Language | 1974 | Foundational paper on turn-taking in conversation. Defines the rules of turn allocation. |
| 25 | Magyari & de Ruiter, "Prediction of Turn-Ends Based on Anticipation of Upcoming Words" | Frontiers in Psychology | 2012 | Humans predict turn-ends ~340ms before they happen. Lexico-syntactic anticipation. |
| 26 | Levinson & Torreira, "Timing in Turn-Taking and its Implications for Processing Models of Language" | Frontiers in Psychology | 2015 | Comprehensive analysis of human turn-taking timing. Gap distribution analysis. |
| 27 | Skantze, "Turn-taking in Conversational Systems and Human-Robot Interaction: A Review" | Computer Speech & Language | 2021 | Survey of computational turn-taking approaches. Covers rule-based to neural methods. |
| 28 | Ward & DeVault, "Ten Challenges in Highly-Interactive Dialog Systems" | AAAI Spring Symposium | 2015 | Identifies key challenges including turn-taking prediction for interactive systems. |

### 10.7 Attention Mechanisms & Model Architecture

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 29 | Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)" | ICLR | 2022 | ALiBi positional encoding. Used in VAP's transformer. Better length generalization. |
| 30 | Arevalo et al., "Gated Multimodal Units for Information Fusion" | ICLR Workshop | 2017 | GMU fusion mechanism. Learns to weight modalities dynamically. |
| 31 | Jaegle et al., "Perceiver: General Perception with Iterative Attention" | ICML | 2021 | Bottleneck attention for multimodal fusion. Relevant to Option C fusion. |

### 10.8 Loss Functions & Training Techniques

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 32 | Lin et al., "Focal Loss for Dense Object Detection" | ICCV | 2017 | Focal loss for class imbalance. γ parameter controls focus on hard examples. |
| 33 | Müller et al., "When Does Label Smoothing Help?" | NeurIPS | 2019 | Analysis of label smoothing benefits. Improves calibration. |

### 10.9 Surveys & Benchmarks

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 34 | Castillo-Lopez et al., "Recent Advances on Turn-taking Modeling" | IWSDS | 2025 | Comprehensive survey. Finding: 72% of works don't compare with previous methods — highlighting benchmarking gap. |
| 35 | Leishman et al., "Multi-stream Turn-taking Prediction" | SemDial | 2024 | Comparison of multi-stream approaches for turn prediction. |
| 36 | "Turn-Taking Review: Technologies, Methods, and Challenges" | MDPI Technologies | 2025 | Broad review covering rule-based to neural turn-taking systems. |

### 10.10 Datasets

| # | Dataset | Source | Details |
|---|---------|--------|---------|
| 37 | Switchboard | LDC | 2,400+ telephone conversations, English. Standard benchmark for turn-taking. |
| 38 | Fisher | LDC | 11,699 conversations, English. Large-scale telephone speech. |
| 39 | AMI Meeting Corpus | Edinburgh | 100+ hours of meeting recordings. Multi-party, multi-modal. |
| 40 | HCRC Map Task | Edinburgh | Task-oriented dialogues with detailed turn annotations. |
| 41 | Switchboard Dialog Act Corpus (SWDA) | Stanford | Switchboard + dialog act annotations (42 acts). |
| 42 | ETD Dataset (Krisp) | Krisp | 120k+ samples, 300+ hours. Ternary: Speaking/Pause/Gap. |

---

## 11. Potential Conference Targets

### 11.1 Top Venues (by relevance)

| Priority | Conference | Deadline (typical) | Focus | Fit |
|----------|-----------|-------------------|-------|-----|
| 1 | **Interspeech** | Mar-Apr | Speech processing | Perfect — VAP papers published here |
| 2 | **ACL / EMNLP** | Jan / May | NLP + multimodal | Good — multimodal + Vietnamese NLP angle |
| 3 | **ICASSP** | Oct | Signal processing + speech | Good — acoustic + speech SSL angle |
| 4 | **EACL / NAACL** | Oct / Dec | NLP | Good — linguistic features for turn-taking |
| 5 | **SIGdial** | Apr-May | Dialogue systems | Perfect — specialized dialogue venue |
| 6 | **LREC-COLING** | varies | Language resources | Good — Vietnamese language resource angle |
| 7 | **IWSDS** | varies | Spoken dialogue | Perfect — turn-taking is core topic |
| 8 | **ICMI** | May | Multimodal interaction | Good — multimodal fusion angle |

### 11.2 Paper Framing Options

**Option A: Speech/Audio venue (Interspeech, ICASSP)**
> "MM-VAP-VI: Multimodal Voice Activity Projection for Vietnamese Turn-Taking Prediction"
> Focus: SSL acoustic features + VAP framework + Vietnamese-specific challenges (tonal language)

**Option B: NLP venue (ACL, EMNLP)**
> "Integrating Vietnamese Discourse Markers and Pre-trained Language Models for Turn-Taking Prediction in Spoken Dialogue"
> Focus: PhoBERT + HuTuDetector + multimodal fusion + Vietnamese linguistics

**Option C: Dialogue venue (SIGdial, IWSDS)**
> "Low-Latency Turn-Taking Prediction for Vietnamese Human-AI Conversation via Multimodal Voice Activity Projection"
> Focus: Latency reduction + human-AI application + streaming architecture

**Option D: Multilingual/Resource (LREC-COLING)**
> "First Multimodal Turn-Taking Prediction System for Vietnamese: Dataset, Model, and Analysis"
> Focus: Vietnamese resource creation + cross-lingual comparison + language-specific analysis
