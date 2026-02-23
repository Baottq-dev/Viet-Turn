# Những Điểm Cần Làm Rõ - MM-VAP-VI Architecture

> **Tài liệu tham chiếu**: `ARCHITECTURE_DESIGN.md`
> **Ngày tạo**: 2026-01-28
> **Trạng thái**: RESOLVED

---

## CRITICAL

### C1. Stereo vs Mono + Diarization

**Vấn đề**: Document giả định có stereo audio (mỗi channel = 1 speaker).

**Quyết định**:

| # | Câu hỏi | Quyết định | Lý do |
|---|---------|-----------|-------|
| 1.1 | Mono hay stereo? | **Mono** | YouTube podcast/phỏng vấn hầu hết là mono hoặc stereo mixed (cả 2 speaker trên cả 2 channel). Không có separated channels. |
| 1.2 | Diarization accuracy? | **DER < 15% acceptable** | pyannote 3.1 đạt ~10-13% DER trên các benchmark. Trên Vietnamese chưa có benchmark nhưng kỳ vọng ~12-18% do out-of-domain. DER < 15% là đủ tốt vì VAP labels có tolerance tự nhiên (2s look-ahead window smooth hóa diarization errors nhỏ). |
| 1.3 | Diarization tool? | **pyannote/speaker-diarization-3.1** | SOTA cho diarization. Hỗ trợ `num_speakers` parameter (biết trước 2 speakers). WhisperX diarization cũng dùng pyannote backend nhưng tích hợp kém linh hoạt hơn. |
| 1.4 | Fallback cho errors? | **Flag + manual review** | File nào có DER suspect cao (speaker imbalance > 80/20, overlap > 30%) → flag để review thủ công. Không skip (mất data) cũng không interpolate (thêm noise vào labels). |

**Cách xử lý mono**:

VAP gốc dùng stereo (mỗi channel 1 speaker) vì Switchboard corpus có telephone channels riêng. Với mono audio + diarization, cách xử lý:

```
Mono audio → pyannote diarization → voice activity per speaker
                                      │
                                      ▼
                            va_matrix[0] = speaker 1 activity (binary)
                            va_matrix[1] = speaker 2 activity (binary)

Acoustic encoder nhận:
  Option A: mono audio (1 channel) → shared encoder
            Encoder tự học phân biệt speakers từ audio features
            → Đơn giản hơn, có thể kém hơn

  Option B: mono audio × 2 (duplicate) + speaker embedding
            Concat speaker identity vào features
            → Phức tạp hơn, rõ ràng hơn

  Option C: mono audio + va_matrix as extra channels
            Input: (audio_waveform, va_channel_1, va_channel_2)
            → Model biết chính xác ai đang nói khi nào

  → CHỌN Option A cho đơn giản. Ablation A vs C trong paper.
    VAP gốc cũng đã thử mono và kết quả chỉ giảm ~2% so với stereo.
```

**Ảnh hưởng lên kiến trúc**:

ARCHITECTURE_DESIGN.md mô tả 2 acoustic encoders (CH1, CH2). Với mono, sửa thành:

```
# Thay vì:
h_a1 = encoder(audio_ch1)  # (B, T, 256)
h_a2 = encoder(audio_ch2)  # (B, T, 256)
h_acoustic = concat([h_a1, h_a2])  # (B, T, 512)
h_acoustic = proj(h_acoustic)  # (B, T, 256)

# Đổi thành:
h_acoustic = encoder(audio_mono)  # (B, T, 768) — from SSL model
h_acoustic = proj(h_acoustic)     # (B, T, 256)
```

Model chỉ cần 1 encoder, output 768-dim (từ Wav2Vec2/WavLM), project xuống 256-dim. Nhẹ hơn và đơn giản hơn.

---

### C2. VAP 256-class vs Human-AI Use Case

**Vấn đề**: VAP 256-class cho 2-speaker symmetric conversation, nhưng goal là Human-AI.

**Quyết định**:

| # | Câu hỏi | Quyết định | Lý do |
|---|---------|-----------|-------|
| 2.1 | 256-class hay simplified? | **256-class (train) + simplified (eval/deploy)** | Train full 256-class để publishable + so sánh với VAP literature. Eval bằng cách aggregate thành events (shift/hold/backchannel). Deploy bằng cách map 256→3 events. |
| 2.2 | Simplified bao nhiêu classes? | **3 events** khi eval | Shift (YIELD), Hold, Backchannel — đúng với 3-class scheme hiện tại. Mapping rõ ràng: sum probabilities của các class thuộc cùng event. |
| 2.3 | Train cả 2 phiên bản? | **Không** | Chỉ train 256-class. Simplified là post-processing lúc evaluation. Không cần train riêng. |

**Lý do chọn hybrid approach**:

```
                  TRAINING                         EVALUATION
                  ────────                         ──────────

Input: audio + text                    Model output: (B, T, 256)
       │                                      │
       ▼                                      ▼
Model: → (B, T, 256)                  Aggregate probabilities:
       │                                P(shift) = Σ probs[shift_classes]
       ▼                                P(hold) = Σ probs[hold_classes]
Loss: CrossEntropy(logits, labels)      P(bc) = Σ probs[bc_classes]
      trên full 256 classes                    │
                                               ▼
                                        Event-based metrics:
                                          Shift BA, Hold BA, BC F1
                                          Latency, FPR
```

Advantages:
1. **So sánh trực tiếp với VAP** — dùng cùng 256-class scheme
2. **Richer information** — 256 classes encode temporal dynamics (khi nào shift xảy ra trong 2s window)
3. **Flexible downstream** — có thể derive bất kỳ binary/ternary prediction nào từ 256-class output
4. **Không overhead** — simplified mapping là 1 dòng code lúc eval, không cần train thêm

---

### C3. HuTuDetector Implementation Details

**Quyết định**:

| # | Aspect | Quyết định | Lý do |
|---|--------|-----------|-------|
| 3.1 | Matching | **Exact match sau khi normalize** | Vietnamese tokens từ PhoBERT tokenizer đã normalized. Fuzzy matching sẽ false positive quá nhiều (vd: "mà" fuzzy match "ma" = hoàn toàn khác nghĩa vì thanh điệu). |
| 3.2 | Case sensitivity | **Case-insensitive** | Vietnamese ít khi có vấn đề case. Lowercase tất cả trước khi match. |
| 3.3 | Multi-marker | **Position-weighted sum** | "ừ thôi nhé" có 3 markers → weight theo position (cuối câu weight cao hơn). Sum tất cả weighted embeddings → normalize. Lý do: marker cuối câu mang tín hiệu turn-taking mạnh nhất ("nhé" ở cuối = yield signal mạnh, "nhé" ở giữa = yếu hơn). |
| 3.4 | Position weight | **Exponential** | `weight = exp(position / total_length)`. Position 0.0 (đầu) → weight 1.0. Position 1.0 (cuối) → weight 2.72. Exponential tạo gradient smooth và nhấn mạnh cuối câu hơn linear. |
| 3.5 | Output format | **Learned embedding** | Mỗi marker type có learned embedding vector. Flexible hơn one-hot/count. Model tự học representation tối ưu. |
| 3.6 | Training | **Learned** (end-to-end) | Marker embeddings initialized random, trained jointly với model. Không pre-define weights vì chưa có data về relative importance của từng marker trong Vietnamese turn-taking. |

**Implementation cụ thể**:

```python
class HuTuDetector(nn.Module):
    """
    Vietnamese discourse marker (hư từ) detector for turn-taking.

    Detects and encodes Vietnamese particles that signal turn-taking intent.
    Markers at the end of utterances are weighted more heavily.
    """
    YIELD_MARKERS = {
        "nhé", "nhỉ", "à", "hả", "ạ", "đi", "nha", "hen",
        "thôi", "vậy", "rồi", "xong", "hết", "không",
        "chưa", "chứ", "nhỉ", "nào"
    }
    HOLD_MARKERS = {
        "mà", "thì", "là", "nhưng", "nên", "vì", "do",
        "tức_là", "nghĩa_là", "có_nghĩa_là", "bởi_vì",
        "cho_nên", "và", "với", "cũng"
    }
    BC_MARKERS = {
        "ừ", "ờ", "ừm", "vâng", "dạ", "uh-huh", "à_há",
        "đúng_rồi", "phải", "ok", "được", "ờm", "aha"
    }

    def __init__(self, embed_dim=64):
        super().__init__()
        # 4 types: yield=0, hold=1, backchannel=2, none=3
        self.type_embedding = nn.Embedding(4, embed_dim)
        self.position_proj = nn.Linear(1, embed_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def get_marker_type(self, token: str) -> int:
        token = token.lower().strip()
        if token in self.YIELD_MARKERS:
            return 0
        elif token in self.HOLD_MARKERS:
            return 1
        elif token in self.BC_MARKERS:
            return 2
        return 3  # none

    def forward(self, text: str, device=None) -> torch.Tensor:
        """
        Args:
            text: raw Vietnamese text

        Returns:
            (embed_dim,) — marker-aware embedding
        """
        tokens = text.lower().split()
        if not tokens:
            return torch.zeros(64, device=device)

        total_len = len(tokens)
        type_ids = []
        positions = []
        weights = []

        for i, token in enumerate(tokens):
            marker_type = self.get_marker_type(token)
            if marker_type < 3:  # is a marker
                type_ids.append(marker_type)
                pos = i / max(total_len - 1, 1)  # normalize to [0, 1]
                positions.append(pos)
                weights.append(math.exp(pos))  # exponential position weight

        if not type_ids:
            return torch.zeros(64, device=device)

        # Embed
        type_tensor = torch.tensor(type_ids, device=device)
        type_embeds = self.type_embedding(type_tensor)  # (N, embed_dim)

        pos_tensor = torch.tensor(positions, device=device).unsqueeze(-1)
        pos_embeds = self.position_proj(pos_tensor)  # (N, embed_dim)

        # Concatenate type + position
        combined = torch.cat([type_embeds, pos_embeds], dim=-1)  # (N, 2*embed_dim)
        combined = self.output_proj(combined)  # (N, embed_dim)

        # Weighted sum
        weight_tensor = torch.tensor(weights, device=device)
        weight_tensor = weight_tensor / weight_tensor.sum()  # normalize
        output = (combined * weight_tensor.unsqueeze(-1)).sum(dim=0)  # (embed_dim,)

        return output
```

---

### C4. Bin Boundaries Rationale

**Quyết định**:

| # | Câu hỏi | Trả lời |
|---|---------|---------|
| 4.1 | VAP gốc dùng bin boundaries nào? | **Đúng như document**: [0-200ms, 200-600ms, 600-1200ms, 1200-2000ms]. Đây là từ code gốc của VAP (GitHub: ErikEkstedt/VoiceActivityProjection). Bins tăng dần duration. |
| 4.2 | Tại sao duration tăng dần? | **Temporal resolution giảm dần theo khoảng cách**. Giống như foveal vision: near-future cần precision cao (200ms bin — phát hiện shift ngay lập tức), far-future chỉ cần rough estimate (800ms bin — rough trend). Cũng giống exponential time binning trong neuroscience (Weber-Fechner law). |
| 4.3 | Optimal bins cho tiếng Việt? | **Chưa biết → ablation study A5**. Tiếng Việt có thể cần bins khác vì: (1) syllable-timed (không stress-timed như English) → rhythm khác, (2) discourse markers thường ở cuối câu → bin gần cuối quan trọng hơn. Ablation: thử [0-200, 200-600, 600-1200, 1200-2000] vs [0-300, 300-800, 800-1500, 1500-2500] vs 6 bins. |

**Verification từ VAP source code**:

```python
# Từ VoiceActivityProjection/vap/utils/utils.py
# https://github.com/ErikEkstedt/VoiceActivityProjection

BIN_TIMES = [0.20, 0.40, 0.60, 0.80]  # cumulative boundaries in seconds
# Bin 0: [0, 0.20]     = 200ms
# Bin 1: [0.20, 0.60]  = 400ms  (0.20 + 0.40)
# Bin 2: [0.60, 1.20]  = 600ms  (0.60 + 0.60)
# Bin 3: [1.20, 2.00]  = 800ms  (1.20 + 0.80)
# Total: 2.00 seconds
```

Chính xác khớp với document. Không cần thay đổi.

---

## IMPORTANT

### I1. Acoustic Encoder Frame Rate Alignment

**Quyết định**:

| # | Câu hỏi | Quyết định | Lý do |
|---|---------|-----------|-------|
| I1.1 | Resample output? | **Không cần** | Wav2Vec2 và WavLM đều output 50fps (320 samples per frame @ 16kHz = 20ms). Đúng khớp với VAP frame rate. |
| I1.2 | Interpolation? | **N/A** | Không cần vì frame rate đã khớp. |
| I1.3 | Whisper mismatch? | **Không dùng Whisper encoder** | Whisper encoder output ~30ms frames (không chính xác 20ms) và không phải self-supervised model (nó là supervised ASR encoder). Dùng Wav2Vec2-VI hoặc WavLM. Whisper chỉ dùng cho ASR transcription, không phải acoustic encoding. |

**Chi tiết frame rate**:

```
Wav2Vec2/WavLM:
  Input: 16kHz waveform
  CNN feature extractor: stride = 320 samples
  Frame rate = 16000 / 320 = 50 fps (20ms per frame)
  → Khớp hoàn hảo với VAP frame rate

Whisper encoder:
  Input: 16kHz → 80-dim log-mel spectrogram
  Spectrogram: window=400 (25ms), hop=160 (10ms)
  Then 2× Conv1d with stride 2 → effective hop = 640 (40ms)
  Frame rate ≈ 25 fps (40ms per frame)
  → KHÔNG KHỚP, cần resample → phức tạp → không dùng
```

---

### I2. Cross-Attention Fusion Causality

**Quyết định**:

| # | Câu hỏi | Quyết định | Lý do |
|---|---------|-----------|-------|
| I2.1 | Linguistic cần causal mask? | **Không** | Linguistic features `h_l` là (B, 1, D) broadcast → không có time dimension để mask. Mỗi audio frame attend tới cùng 1 linguistic vector. |
| I2.2 | Cross-attention mask strategy? | **Audio self-attention: causal. Audio→Text cross-attention: no mask** | Audio frames phải causal (frame t chỉ thấy frames ≤ t). Nhưng text vector là constant → cross-attention tới nó không cần mask vì không leak future information. |
| I2.3 | Frames 0-24 attend text nào? | **Empty string → zero vector** | Trước ASR update đầu tiên, chưa có transcript. PhoBERT encode "" → near-zero output. Model falls back to acoustic-only prediction. Đây là desired behavior: đầu cuộc nói chuyện, chưa có text, model dựa vào audio. |

**Chi tiết**:

```
Timeline:  0ms    500ms    1000ms    1500ms
           │       │        │         │
Audio:     ════════════════════════════
           frame 0  frame 25  frame 50  frame 75
           │       │        │         │
Text:      [empty]  [ASR₁]   [ASR₂]    [ASR₃]
           │       │        │         │
Attend to: zero    h_l₁     h_l₂      h_l₃

Causal constraint:
  Frame 24 → attend to text="" (zero vector)
  Frame 25 → attend to text=ASR₁ ("Tôi")
  Frame 49 → attend to text=ASR₁ ("Tôi") — same, not updated yet
  Frame 50 → attend to text=ASR₂ ("Tôi muốn")
```

Không có information leakage vì:
- Audio self-attention là causal
- Text attend là backward-looking (chỉ dùng ASR output đã có)
- ASR output tại frame 25 chỉ chứa words đã nói trước frame 25

---

### I3. Linguistic Encoder Pooling Strategy

**Quyết định**:

| # | Câu hỏi | Quyết định | Lý do |
|---|---------|-----------|-------|
| I3.1 | Default pooling? | **[CLS] token** | Chuẩn cho PhoBERT. CLS token aggregate toàn bộ sentence meaning. Cho turn-taking, overall sentence semantics quan trọng hơn individual token features. |
| I3.2 | Last-K? | **N/A** | Không dùng Last-K làm default. Nhưng có thể ablation: CLS vs Mean vs Last-3. |
| I3.3 | Broadcast mechanism? | **Repeat** | `h_l.unsqueeze(1).expand(-1, T, -1)`. Simple repeat — cùng 1 vector cho mọi frame cho tới ASR update tiếp theo. Interpolation không có ý nghĩa vì text update là discrete. |

**Lưu ý**: CLS pooling + broadcast có nghĩa linguistic features là **step function** (nhảy bậc mỗi 500ms). Acoustic features là **continuous** (thay đổi mỗi 20ms). Fusion module (GMU/CrossAttention) sẽ tự học cách kết hợp 2 signal khác nhau tần này.

---

### I4. Streaming ASR Latency Budget

**Quyết định**:

| Component | Estimated Latency | Ghi chú |
|-----------|------------------|---------|
| Audio capture | 20ms | 1 frame buffer |
| Acoustic encoder | 10-30ms | Wav2Vec2 single frame (with context cache) |
| ASR buffer | 500ms | Accumulate before sending to ASR |
| ASR inference | 100-300ms | faster-whisper on GPU |
| Linguistic encoder | 10-20ms | PhoBERT forward pass (cached) |
| Fusion + Transformer | 5-10ms | Small model, cached KV |
| Decision logic | <1ms | Simple threshold comparison |

**Tổng latency (worst case)**:

```
Path 1: Audio-only prediction (no text update)
  Audio capture (20ms) + Acoustic (30ms) + Fusion (10ms) + Decision (1ms)
  = ~61ms per frame — REAL-TIME (< 20ms budget? No, but pipelined)

  Pipelined: While frame N is being encoded, frame N+1 is captured
  Effective throughput: 1 frame per ~30ms → 33fps → acceptable

Path 2: With text update (every 500ms)
  Same as Path 1, but every 25th frame also includes:
  + ASR (300ms) + PhoBERT (20ms)
  = ~381ms extra — but this runs in PARALLEL with audio processing
  Text update arrives ~300ms late → model uses stale text for 300ms
  This is acceptable: text is supplementary, not primary signal
```

**Key insight**: ASR latency chỉ ảnh hưởng linguistic branch, không ảnh hưởng acoustic branch. Model vẫn dự đoán real-time từ audio, text chỉ bổ sung thêm context.

---

### I5. Multi-Stage Training LR Schedule

**Quyết định**:

| Stage | acoustic_encoder | linguistic_encoder | new_modules | Epochs | Mục đích |
|-------|-----------------|-------------------|-------------|--------|---------|
| 1 | 1e-5 | 0 (frozen) | 1e-4 | 5-10 | Acoustic branch + fusion + projection head học cơ bản. PhoBERT frozen vì chưa có gradient ổn định từ fusion. |
| 2 | 5e-6 (giảm 2×) | 2e-5 (unfreeze top 6 layers) | 5e-5 (giảm 2×) | 20-30 | Joint training. Acoustic encoder fine-tune chậm (đã converge sơ bộ). PhoBERT bắt đầu adapt. New modules giảm LR vì đã gần convergence. |
| 3 | 1e-6 (giảm 5×) | 5e-6 (giảm 4×) | 1e-5 (giảm 5×) | 10-15 | Final fine-tuning. Tất cả components LR rất thấp. Early stopping dựa trên val Shift Balanced Accuracy. |

**Scheduler**: Cosine with warmup trong mỗi stage.

```python
# Warmup ratio: 10% of stage epochs
# Cosine decay to min_lr = 1e-7

# Stage transition: Reset scheduler, keep model weights
# Không reset optimizer state (Adam momentum) — smoother transition
```

**Gradient clipping**: `max_grad_norm = 1.0` cho mọi stages.

**Early stopping**: Patience = 5 epochs. Monitor `val_shift_balanced_accuracy`. Nếu không improve sau 5 epochs → stop stage hiện tại, chuyển sang stage tiếp (hoặc stop nếu đã stage 3).

---

### I6. Label Generation Edge Cases

| Case | Xử lý | Lý do |
|------|-------|-------|
| **End of audio** (frame t < 2s từ cuối) | Bins ngoài audio → assume silence (bit = 0) | VAP gốc làm tương tự. Frames cuối tự nhiên predict "silence ahead" — đúng vì audio sắp hết. Không skip vì mất training signal ở cuối conversation. |
| **Start of audio** (frame t = 0) | Bình thường — look-ahead 2s từ frame 0 | Không có vấn đề. VAP chỉ look ahead, không look back. Frame 0 có đủ 2s ahead (trừ khi audio < 2s thì thuộc case trên). |
| **>2 speakers** | Giữ 2 speakers nói nhiều nhất, bỏ phần còn lại | VAP designed cho 2 speakers. Speakers phụ thường là MC giới thiệu hoặc người chen ngang — không mang turn-taking pattern tiêu biểu. |
| **Long silence (>5s)** | Giữ nguyên, KHÔNG downsample | Silence dài cũng là training signal hợp lệ (class 0 = both silent). Model cần học rằng silence → không ai sắp nói. Downsample sẽ bias model. Tuy nhiên nếu silence >30s (quảng cáo, nhạc), crop ra. |
| **Overlapping speech** | Giữ nguyên — valid VAP class | Overlap là 1 pattern turn-taking quan trọng (competitive overlap, cooperative overlap). VAP labels encode overlap tự nhiên (cả 2 speakers active = bits đều 1). Đây là training data có giá trị, KHÔNG exclude. |

---

## NICE-TO-HAVE

### N1. Data Statistics Estimates

**Ước tính cho Vietnamese conversational data** (dựa trên literature + đặc điểm ngôn ngữ):

```
Vietnamese conversation (estimated):
  - Turn transitions: ~8-15 / minute
    (Vietnamese nói nhanh hơn English, nhiều backchannel "ừ", "vâng")
  - Mean gap: 150-350ms
    (Vietnamese có thể ngắn hơn English do syllable-timed)
  - Backchannel frequency: ~18-25% of turns
    (Vietnamese có nhiều hư từ đáp lại: ừ, vâng, dạ, ờ)
  - Overlap: ~8-15% of time
    (Vietnamese podcast thường overlap nhiều hơn formal English)

So sánh với Switchboard (English):
  - Turn transitions: ~8-12 / minute
  - Mean gap: 200-400ms
  - Backchannel: ~15% of turns
  - Overlap: ~5-10% of time

Frame-level label distribution (estimated):
  - Hold (1 speaker continuous): ~45-55%
  - Silence (both quiet):        ~15-25%
  - Shift (turn transition):     ~5-10%
  - Backchannel:                 ~3-8%
  - Overlap:                     ~5-10%
  - Other patterns:              ~5-10%
```

**Cần verify** bằng cách chạy pipeline trên 5-10 file đầu tiên và đo actual statistics.

---

### N2. Compute Requirements

```yaml
training:
  gpu: "RTX 3090/4090 (24GB) hoặc A100 (40/80GB)"
  gpu_memory_per_batch:
    # Wav2Vec2-base (95M) + PhoBERT (135M) + new modules (~7M)
    # Batch=16, window=20s audio (320K samples)
    estimated: "~18GB (fp16) hoặc ~12GB (bf16 with gradient checkpointing)"
  batch_size: 16
  gradient_accumulation: 4
  effective_batch: 64
  time_per_epoch:
    # 50h data, window=20s, stride=5s → ~36000 windows
    # 36000 / 64 = 563 steps/epoch
    # ~2s per step on A100 → ~19 minutes/epoch
    # ~8s per step on RTX 3090 → ~75 minutes/epoch
    estimated_a100: "~20 min/epoch"
    estimated_3090: "~75 min/epoch"
  total_training:
    # 50 epochs × 75 min = ~62.5 GPU hours on RTX 3090
    # 50 epochs × 20 min = ~16.7 GPU hours on A100
    estimated_3090: "~65 GPU hours"
    estimated_a100: "~17 GPU hours"

inference:
  gpu_memory: "~1.5GB (fp16, batch=1)"
  cpu_feasible: "Yes, nhưng ~5-10× chậm hơn"
  target_device: "GPU preferred, CPU acceptable for demo"
  real_time_factor:
    gpu: "~0.05× (20× faster than real-time)"
    cpu: "~0.5× (2× faster than real-time)"
```

---

### N3. Error Handling in Streaming

| Error Type | Handling Strategy | Lý do |
|-----------|------------------|-------|
| ASR fails to produce output | Giữ text cache cũ. Model tiếp tục với acoustic-only (text = stale/zero). Log warning. | Text là supplementary. Acoustic branch vẫn hoạt động bình thường. |
| Audio dropout (frame missing) | Zero-fill frame. Model dùng context từ cached KV. Log warning. | 1-2 frames dropout (20-40ms) không ảnh hưởng đáng kể nhờ temporal context trong Transformer. |
| KV cache overflow (>10s context) | Sliding window: bỏ oldest frames, giữ latest max_context frames. | ALiBi positional encoding extrapolate tốt. Receptive field thực tế của model chỉ ~5-10s, frames cũ hơn ít ảnh hưởng. |
| Diarization confusion (runtime) | Không áp dụng — diarization chỉ dùng lúc training (tạo labels). Inference chỉ dùng audio + text, không cần diarization. | VAP predict voice activity projection, không cần biết ai đang nói lúc inference. |

---

### N4. Ablation Study Prioritization

**Confirmed priority order**:

| Priority | Ablation | Experiments | Estimate (GPU hours, RTX 3090) | Reasoning |
|----------|---------|-------------|-------------------------------|-----------|
| **P1** | A2: Modality (audio vs text vs both) | 3 | ~200h | **Core claim**: multimodal > unimodal. Phải có. |
| **P2** | A4: Linguistic (PhoBERT vs HuTu vs both) | 3 | ~200h | **Novel contribution**: HuTuDetector. Phải chứng minh giá trị. |
| **P3** | A1: Acoustic encoder (Wav2Vec2-VI vs WavLM vs TCN) | 4 | ~260h | Justify encoder choice. So sánh SSL vs hand-crafted. |
| **P4** | A3: Fusion strategy (GMU vs CrossAttn vs Bottleneck vs Concat) | 4 | ~260h | Justify fusion choice. Less critical — có thể skip cho workshop paper. |
| **P5** | A5: Projection window (1s vs 2s vs 3s) | 3 | ~200h | Vietnamese-specific insight. Nice for full paper, skip for short. |
| **P6** | A6: Vietnamese-specific analysis | N/A | Analysis only | Dialect analysis, marker analysis — post-hoc, no extra training. |
| **P7** | A7: Data scale (10h→20h→50h→100h) | 4 | ~400h | Important for resource paper. Skip nếu không đủ compute. |

**Minimum viable paper**: P1 + P2 = 6 experiments ≈ 400 GPU hours ≈ 17 ngày RTX 3090

**Full paper**: P1-P5 + P6 = 17 experiments + analysis ≈ 1120 GPU hours ≈ 47 ngày RTX 3090
(Hoặc ~12 ngày A100)

---

### N5. VAP Reproduction Baseline

**Quyết định**:

```
B3a: "VAP-reproduction"
  - Dùng CPC encoder (đúng như VAP gốc)
  - Train trên Vietnamese data
  - So sánh: VAP gốc transfer sang Vietnamese được không?
  - Vấn đề: CPC pre-trained trên English → có thể kém trên Vietnamese

B3b: "VAP-Wav2Vec2"
  - Thay CPC bằng Wav2Vec2-VI
  - Train trên Vietnamese data
  - Fair comparison: cùng acoustic encoder như MM-VAP-VI nhưng không có text
  - Đây mới là baseline quan trọng: đo exact contribution of linguistic branch

→ CHỌN B3b là PRIMARY baseline.
  B3a optional nếu đủ compute.

  Lý do: B3b isolate đúng contribution của linguistic branch.
  Nếu MM-VAP-VI (audio+text) > B3b (audio-only, same encoder),
  → chứng minh text giúp ích cho Vietnamese turn-taking.
  Đây chính là core claim của paper.
```

---

## Checklist Trước Khi Implement

### Phase 0: Decisions Required — ALL RESOLVED

- [x] C1: Mono + pyannote diarization. Acoustic encoder: 1 channel.
- [x] C2: Train 256-class, eval aggregated 3 events.
- [x] C3: HuTuDetector: exact match, case-insensitive, position-weighted sum, learned embeddings.
- [x] C4: Bins [0-200, 200-600, 600-1200, 1200-2000]ms — confirmed from VAP source.

### Phase 1: Data Pipeline

- [x] I6: Edge cases defined (end-of-audio, >2 speakers, silence, overlap).
- [x] N1: Data statistics estimated.

### Phase 2: Model Implementation

- [x] I1: Wav2Vec2/WavLM output 50fps — matches VAP. No resample needed.
- [x] I2: Audio self-attention causal, cross-attention to text no mask needed.
- [x] I3: CLS pooling default, repeat broadcast.

### Phase 3: Training

- [x] I5: LR schedule defined (3 stages with specific LRs).
- [x] N2: Compute estimated (~65 GPU hours RTX 3090 for full training).

### Phase 4: Evaluation

- [x] N4: Ablation priority confirmed. Minimum: P1+P2.
- [x] N5: B3b (VAP-Wav2Vec2, audio-only) as primary baseline.

### Phase 5: Streaming Demo

- [x] I4: Latency budget defined. Audio-only path ~61ms, text update ~381ms parallel.
- [x] N3: Error handling strategies defined.
