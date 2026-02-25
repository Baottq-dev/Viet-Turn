# MM-VAP-VI: Danh sách thí nghiệm chi tiết

> **Mục đích:** Liệt kê tất cả runs cần thực hiện, thứ tự ưu tiên, và trạng thái.
> **Target:** INTERSPEECH / ACL Findings / SIGdial Workshop.

---

## Tổng quan

| Nhóm | Số runs | GPU time (ước tính) | Ưu tiên |
|-------|---------|--------------------:|---------|
| Baselines (không train) | 4 | 0h | P0 |
| Incremental Ablation (B0–B4) | 4 + 3 seeds | ~68h | P0 |
| Lateral Ablation (L1–L4) | 4 | ~48h | P1 |
| Analysis (trên checkpoint B4) | 5 | ~4h | P1 |
| Validation (manual) | 3 | 0h GPU | P2 |
| **Tổng** | **~20 runs** | **~120h** | |

> GPU time ước tính trên single A100/RTX 4090. Thực tế phụ thuộc vào hardware.

---

## PHASE 0 — Baselines không cần train [ƯU TIÊN P0]

Chạy ngay được, không cần GPU. Cho kết quả lower bound để so sánh.

### RUN-01: Random Baseline

- **Mô tả:** Dự đoán ngẫu nhiên 256 classes
- **Config:** Không cần — chỉ cần code
- **Kết quả kỳ vọng:** ~0.4% frame accuracy
- **Lệnh:**
```python
# Trong evaluate.py hoặc notebook
import numpy as np
rng = np.random.RandomState(42)
random_preds = rng.randint(0, 256, size=len(valid_labels))
acc = (random_preds == valid_labels).mean()
```

### RUN-02: Majority Class Baseline

- **Mô tả:** Luôn dự đoán class phổ biến nhất (thường class 0 = silence)
- **Config:** Không cần
- **Kết quả kỳ vọng:** ~30-50% accuracy
- **Lệnh:**
```python
from collections import Counter
majority = Counter(train_labels.tolist()).most_common(1)[0][0]
acc = (test_labels == majority).mean()
```

### RUN-03: Silence-Threshold Baseline

- **Mô tả:** Rule-based: dự đoán SHIFT sau N ms im lặng liên tục.
  Chạy với nhiều ngưỡng: 200ms, 500ms, 700ms, 1000ms.
  **Đây là baseline bắt buộc** — mọi paper VAP đều có.
- **Config:** Không cần train
- **Kết quả kỳ vọng:** Shift/Hold BA ~0.55-0.60 tùy ngưỡng
- **Cần implement:** `scripts/silence_threshold_baseline.py`
- **Logic:**
```python
def silence_threshold_baseline(va_matrix, threshold_ms, frame_hz=50):
    """Predict SHIFT if speaker silent for >= threshold_ms."""
    threshold_frames = int(threshold_ms / (1000 / frame_hz))
    s0 = va_matrix[0].numpy()
    silence_count = 0
    pred_shift = np.zeros(len(s0))
    for t in range(len(s0)):
        if s0[t] == 0:
            silence_count += 1
            if silence_count >= threshold_frames:
                pred_shift[t] = 1.0
        else:
            silence_count = 0
    return pred_shift
```

### RUN-04: Text-only Baseline

- **Mô tả:** Chỉ dùng PhoBERT + HuTu, không có audio.
  Cho thấy text đóng góp bao nhiêu khi dùng độc lập.
- **Config:** `configs/config_text_only.yaml`
- **Training:** 1-stage, 50 epochs
- **Kết quả kỳ vọng:** ~40-55% frame accuracy
- **Cần implement:** Class `TextOnlyVAP` hoặc config đặc biệt
- **GPU time:** ~8h

---

## PHASE 1 — Incremental Ablation [ƯU TIÊN P0]

**Đây là bảng kết quả quan trọng nhất trong paper.**
Bắt đầu từ bản gốc (Original VAP), thêm từng đóng góp.
B1–B3 dùng **1-stage training** để isolate đóng góp component.
B4 thêm 3-stage training như đóng góp cuối cùng.

```
B0 (Original VAP) → B1 (+Vi encoder) → B2 (+Text) → B3 (+HuTu) → B4 (+3-stage)
```

### RUN-B0: Original VAP (HuBERT-EN, audio-only, 1-stage)

- **Mô tả:** VAP gốc theo Ekstedt & Skantze (2022), chạy trên data tiếng Việt.
  Đây là **bản gốc** để so sánh tất cả cải tiến.
- **Thay đổi so với full model:**
  - Encoder: `facebook/hubert-base-ls960` (thay Wav2Vec2-Vi)
  - Không text, không fusion, không HuTu
  - Training: 1-stage, 50 epochs
- **Config:** `configs/config_b0_original.yaml`
- **Output:** `outputs/b0_original_vap/`
- **Kết quả kỳ vọng:** ~60% frame accuracy, Shift/Hold BA ~0.58
- **GPU time:** ~8h

```yaml
# configs/config_b0_original.yaml — key differences
acoustic_encoder:
  pretrained: "facebook/hubert-base-ls960"
linguistic_encoder:
  phobert: { enabled: false }
  hutu_detector: { enabled: false }
fusion:
  type: "none"
training:
  stage1:
    epochs: 50
    freeze: ["linguistic_encoder", "fusion"]
    lr: { acoustic_encoder: 1.0e-4, transformer: 1.0e-4, projection_head: 1.0e-4 }
  stage2: { epochs: 0 }
  stage3: { epochs: 0 }
```

```bash
python train.py --config configs/config_b0_original.yaml \
    --output-dir outputs/b0_original_vap
```

### RUN-B1: + Vietnamese Encoder (Wav2Vec2-Vi, audio-only, 1-stage)

- **Mô tả:** Thay HuBERT-EN bằng Wav2Vec2-Vi. Vẫn audio-only.
  Chứng minh: **encoder tiếng Việt tốt hơn encoder tiếng Anh**.
- **Thay đổi so với B0:** Chỉ thay encoder
- **Config:** `configs/config_b1_vi_encoder.yaml`
- **Output:** `outputs/b1_vi_encoder/`
- **Kết quả kỳ vọng:** ~71% frame accuracy (+11% so với B0)
- **GPU time:** ~8h

```yaml
# configs/config_b1_vi_encoder.yaml — key differences
acoustic_encoder:
  pretrained: "nguyenvulebinh/wav2vec2-base-vi"  # ← thay đổi duy nhất
# Vẫn không text, không fusion, 1-stage 50ep
```

> **Lưu ý:** B1 tương đương với Audio-only Stage 1, nhưng chạy 50 epochs thay vì 10.

### RUN-B2: + PhoBERT Text (multimodal, no HuTu, 1-stage)

- **Mô tả:** Thêm PhoBERT text branch + Cross-Attention fusion. Chưa có HuTu.
  Chứng minh: **text cải thiện prediction**.
- **Thay đổi so với B1:** Thêm linguistic encoder (PhoBERT only) + fusion
- **Config:** `configs/config_b2_add_text.yaml`
- **Output:** `outputs/b2_add_text/`
- **Kết quả kỳ vọng:** ~73.5% (+2.5% so với B1)
- **GPU time:** ~10h

```yaml
# configs/config_b2_add_text.yaml
acoustic_encoder:
  pretrained: "nguyenvulebinh/wav2vec2-base-vi"
linguistic_encoder:
  phobert: { pretrained: "vinai/phobert-base-v2", enabled: true }
  hutu_detector: { enabled: false }  # ← chưa có HuTu
fusion:
  type: "cross_attention"
training:
  stage1:
    epochs: 50
    freeze: []  # All unfrozen (1-stage)
    lr: { acoustic_encoder: 5.0e-5, linguistic_encoder: 2.0e-5,
          fusion: 1.0e-4, transformer: 5.0e-5, projection_head: 5.0e-5 }
  stage2: { epochs: 0 }
  stage3: { epochs: 0 }
```

### RUN-B3: + HuTu Detector (multimodal, 1-stage)

- **Mô tả:** Thêm HuTu Detector vào text branch.
  Chứng minh: **discourse markers cải thiện turn-taking prediction cho tiếng Việt**.
  Đây là **contribution chính** của paper.
- **Thay đổi so với B2:** Bật HuTu detector
- **Config:** `configs/config_b3_add_hutu.yaml`
- **Output:** `outputs/b3_add_hutu/`
- **Kết quả kỳ vọng:** ~74.5% (+1% so với B2, nhưng +5-10% trên frames có marker)
- **GPU time:** ~10h

```yaml
# configs/config_b3_add_hutu.yaml — chỉ thay đổi 1 dòng so với B2
linguistic_encoder:
  hutu_detector: { enabled: true }  # ← bật HuTu
# Còn lại giống B2
```

### RUN-B4: + 3-Stage Curriculum = Full MM-VAP-VI ⭐

- **Mô tả:** Model đầy đủ với 3-stage training.
  Chứng minh: **curriculum training tốt hơn 1-stage**.
  **Chạy 3 seeds** (42, 123, 456) để báo cáo mean ± std.
- **Thay đổi so với B3:** 3-stage training thay vì 1-stage
- **Config:** `configs/config.yaml` (config chính, đã có sẵn)
- **Output:** `outputs/b4_full_seed42/`, `outputs/b4_full_seed123/`, `outputs/b4_full_seed456/`
- **Kết quả kỳ vọng:** ~75.2% ± 0.5%
- **GPU time:** ~12h × 3 seeds = **36h**

```bash
python train.py --config configs/config.yaml \
    --output-dir outputs/b4_full_seed42 --seed 42
python train.py --config configs/config.yaml \
    --output-dir outputs/b4_full_seed123 --seed 123
python train.py --config configs/config.yaml \
    --output-dir outputs/b4_full_seed456 --seed 456
```

### Bảng kết quả Incremental Ablation (Table 1 trong paper)

```
+------+------------------------+---------+--------+------+--------+------+-------+
|      | Model                  | Frame   | F1     | S/H  | BC     | EoT  | VAQI  |
|      |                        | Acc (%) | (w)    | BA   | F1     | (ms) |       |
+------+------------------------+---------+--------+------+--------+------+-------+
| B0   | Original VAP (HuBERT)  |         |        |      |        |      |       |
| B1   | + Wav2Vec2-Vi          |         |        |      |        |      |       |
| B2   | + PhoBERT text         |         |        |      |        |      |       |
| B3   | + HuTu detector        |         |        |      |        |      |       |
| B4   | + 3-stage (Full) ⭐     |         |        |      |        |      |       |
+------+------------------------+---------+--------+------+--------+------+-------+
  * p < 0.05 vs B0 (paired permutation test)
```

---

## PHASE 2 — Lateral Ablation [ƯU TIÊN P1]

So sánh các lựa chọn thiết kế. Tất cả dùng full model (HuTu + 3-stage).

### RUN-L1: GMU Fusion

- **Mô tả:** Thay Cross-Attention bằng Gated Multimodal Unit
- **So sánh với:** B4 (Cross-Attention)
- **Config:** `configs/config_fusion_gmu.yaml`
- **Output:** `outputs/l1_gmu/`
- **GPU time:** ~12h

```yaml
fusion:
  type: "gmu"
  dim: 256
  dropout: 0.1
```

### RUN-L2: Bottleneck Fusion

- **Mô tả:** Thay Cross-Attention bằng Perceiver-style Bottleneck
- **Config:** `configs/config_fusion_bottleneck.yaml`
- **Output:** `outputs/l2_bottleneck/`
- **GPU time:** ~12h

```yaml
fusion:
  type: "bottleneck"
  dim: 256
  num_heads: 4
  num_latents: 16
  dropout: 0.1
```

### RUN-L3: 2-Stage Training

- **Mô tả:** Audio-only (15ep) → Full (35ep), bỏ qua Stage 3
- **So sánh với:** B4 (3-stage)
- **Config:** `configs/config_2stage.yaml`
- **Output:** `outputs/l3_2stage/`
- **GPU time:** ~12h

```yaml
training:
  stage1: { epochs: 15, freeze: ["linguistic_encoder", "fusion"],
            lr: { acoustic_encoder: 1.0e-4, transformer: 1.0e-4, projection_head: 1.0e-4 } }
  stage2: { epochs: 35, freeze: [],
            lr: { acoustic_encoder: 2.0e-5, linguistic_encoder: 1.0e-5,
                  fusion: 5.0e-5, transformer: 5.0e-5, projection_head: 5.0e-5 } }
  stage3: { epochs: 0 }
```

### RUN-L4: WavLM Encoder

- **Mô tả:** Thay Wav2Vec2-Vi bằng WavLM-base (multilingual)
- **Config:** `configs/config_encoder_wavlm.yaml`
- **Output:** `outputs/l4_wavlm/`
- **GPU time:** ~12h

```yaml
acoustic_encoder:
  type: "wavlm"
  pretrained: "microsoft/wavlm-base"
```

### Bảng kết quả Lateral Ablation (Table 2 trong paper)

```
+------+--------------------------+---------+------+------+
|      | Variant                  | Frame   | S/H  | VAQI |
|      |                          | Acc (%) | BA   |      |
+------+--------------------------+---------+------+------+
| B4   | Full model (Cross-Attn)  |         |      |      |
| L1   | GMU fusion               |         |      |      |
| L2   | Bottleneck fusion        |         |      |      |
| L3   | 2-stage training         |         |      |      |
| L4   | WavLM encoder            |         |      |      |
+------+--------------------------+---------+------+------+
```

---

## PHASE 3 — Analysis [ƯU TIÊN P1]

Chạy trên checkpoint B4 (best seed). Không cần train thêm.

### ANALYSIS-01: Prosody Sensitivity (Pitch Flattening) ⚠️ BẮT BUỘC

- **Mô tả:** Flatten pitch (F0) trong test audio, giữ nguyên nội dung → eval lại B4.
  Tiếng Việt là **ngôn ngữ thanh điệu** → reviewer 100% sẽ hỏi model có dựa vào tone không.
- **Cần implement:** `scripts/flatten_pitch.py`
- **Logic:**
```python
import parselmouth
def flatten_pitch(audio_path, output_path):
    """Flatten F0 to median, keeping content intact."""
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    median_f0 = parselmouth.praat.call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")
    manipulation = parselmouth.praat.call(snd, "To Manipulation", 0.01, 50, 400)
    pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")
    parselmouth.praat.call(pitch_tier, "Remove points between", 0, snd.duration)
    parselmouth.praat.call(pitch_tier, "Add point", snd.duration/2, median_f0)
    parselmouth.praat.call([manipulation, pitch_tier], "Replace pitch tier")
    flat_snd = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")
    flat_snd.save(output_path, "WAV")
```
- **Kết quả kỳ vọng:**
  - Frame Acc giảm ~3-8% (model phụ thuộc vào prosody một phần)
  - Nếu giảm < 2%: model chủ yếu dùng timing, không dùng tone → đề cập trong paper
  - Nếu giảm > 10%: model quá phụ thuộc prosody → cần thảo luận

| Điều kiện | Frame Acc | Shift/Hold BA | Delta |
|-----------|-----------|---------------|-------|
| B4 (bình thường) | ? | ? | — |
| B4 (pitch flattened) | ? | ? | ? |

### ANALYSIS-02: Performance theo Gap Duration

- **Mô tả:** Chia Shift/Hold accuracy theo khoảng im lặng giữa 2 lượt nói.
  Cho thấy model xử lý tốt/kém ở đoạn silence ngắn/dài.
- **Cần implement:** Logic trong evaluation script
- **Bins:** 0-250ms, 250-500ms, 500-750ms, 750ms+

| Gap Duration | Số events | Shift/Hold BA (B0) | Shift/Hold BA (B4) | Delta |
|--------------|-----------|--------------------|--------------------|-------|
| 0–250ms | ? | ? | ? | ? |
| 250–500ms | ? | ? | ? | ? |
| 500–750ms | ? | ? | ? | ? |
| 750ms+ | ? | ? | ? | ? |

### ANALYSIS-03: Data Efficiency / Learning Curve

- **Mô tả:** Train B4 (full model) với 10%, 25%, 50%, 100% training data.
  Quan trọng cho góc "low-resource Vietnamese" — cho thấy cần bao nhiêu data.
- **Config:** Tạo sub-manifests bằng random sampling theo conversation
- **GPU time:** ~6h + 8h + 10h = ~24h (nhưng priority thấp hơn)

| % Data | # Windows | Frame Acc | Shift/Hold BA | VAQI |
|--------|-----------|-----------|---------------|------|
| 10% | ~38 | ? | ? | ? |
| 25% | ~94 | ? | ? | ? |
| 50% | ~188 | ? | ? | ? |
| 100% | 376 | ? | ? | ? |

### ANALYSIS-04: Vietnamese Marker Deep-Dive (Table 3 trong paper)

- **Mô tả:** So sánh accuracy trên frames có/không có discourse markers.
  Dùng code đã có: `src/evaluation/vietnamese_analysis.py`
- **Chạy trên:** Checkpoint B3 (có HuTu) vs B2 (không HuTu)

| Loại Marker | Ví dụ | Count | Acc (B2, no HuTu) | Acc (B3, + HuTu) | Delta |
|-------------|-------|-------|--------------------|--------------------|-------|
| Yield (SFP) | nhé, nhỉ, hả | ? | ? | ? | ? |
| Hold (liên từ) | mà, là, thì | ? | ? | ? | ? |
| Backchannel | ừ, ờ, vâng | ? | ? | ? | ? |
| Turn request | này, ơi | ? | ? | ? | ? |
| Không marker | — | ? | ? | ? | ? |

### ANALYSIS-05: Qualitative Examples

- **Mô tả:** Chọn 3-5 đoạn hội thoại minh họa:
  1. Một ví dụ SHIFT dự đoán đúng (model nhận ra "nhé" → shift)
  2. Một ví dụ HOLD dự đoán đúng (model nhận ra "mà" → hold)
  3. Một ví dụ BACKCHANNEL
  4. Một failure case (model sai) + phân tích tại sao
- **Dùng:** `scripts/visualize_predictions.py` (đã có)
- **Output:** Hình vẽ timeline cho paper (Figure 5-6)

---

## PHASE 4 — Manual Validation [ƯU TIÊN P2]

Không cần GPU. Cần thời gian manual annotation.

### VALIDATION-01: Diarization Error Rate (DER)

- **Mô tả:** Random chọn 20 segments, manually annotate speaker boundaries,
  so sánh với output pyannote (RTTM files).
- **Metric:** DER (%), Missed Speech, False Alarm, Speaker Confusion
- **Thời gian:** ~4-6 giờ manual work
- **Mục đích:** Reviewer sẽ hỏi: *"Annotation pipeline tự động, sao biết chính xác?"*

### VALIDATION-02: ASR Word Error Rate (WER)

- **Mô tả:** Random chọn 20 segments, manually transcribe, so sánh với Whisper output.
- **Metric:** WER (%)
- **Thời gian:** ~6-8 giờ manual work

### VALIDATION-03: Event Annotation Check

- **Mô tả:** Manually label ~100 turn-taking events (shift/hold/BC) trên 10 segments,
  so sánh với automatic classification từ VA matrix.
- **Metric:** Cohen's Kappa, Agreement %
- **Thời gian:** ~4 giờ manual work
- **Mục đích:** Validate rằng classify_events() tạo labels chính xác

---

## Thứ tự chạy theo ưu tiên

### Đợt 1: Core Results (bắt buộc cho paper) — ~80h GPU

```
Tuần 1:
  ├── RUN-01,02,03 (Baselines, không GPU)          → ngay
  ├── RUN-B0 (Original VAP)                         → ~8h
  ├── RUN-B1 (+ Vi encoder)                         → ~8h
  └── RUN-B4-seed42 (Full model, seed chính)        → ~12h

Tuần 2:
  ├── RUN-B2 (+ Text)                               → ~10h
  ├── RUN-B3 (+ HuTu)                               → ~10h
  ├── RUN-B4-seed123 (Full model, seed 2)            → ~12h
  ├── RUN-B4-seed456 (Full model, seed 3)            → ~12h
  └── ANALYSIS-01 (Prosody sensitivity)              → ~2h eval
```

> **Sau đợt 1:** Đã có đủ kết quả cho **Table 1** (incremental ablation) +
> prosody analysis. Có thể bắt đầu viết Method + Results.

### Đợt 2: Ablation + Analysis — ~50h GPU

```
Tuần 3:
  ├── RUN-L1 (GMU fusion)                           → ~12h
  ├── RUN-L2 (Bottleneck fusion)                    → ~12h
  ├── RUN-L3 (2-stage)                              → ~12h
  ├── RUN-04 (Text-only baseline)                   → ~8h
  ├── ANALYSIS-02 (Gap duration)                     → ~1h eval
  └── ANALYSIS-04 (Marker deep-dive)                 → ~1h eval

Tuần 4:
  ├── RUN-L4 (WavLM encoder)                        → ~12h
  ├── ANALYSIS-05 (Qualitative examples)             → ~1h
  └── Bootstrap CI + Permutation tests               → ~2h eval
```

> **Sau đợt 2:** Đã có đủ kết quả cho **Table 2** (lateral ablation) +
> **Table 3** (marker impact) + tất cả figures.

### Đợt 3: Nice-to-have (nếu còn thời gian) — ~30h GPU

```
Tuần 5:
  ├── ANALYSIS-03 (Data efficiency: 10/25/50%)       → ~24h
  ├── VALIDATION-01,02,03 (Manual annotation)        → manual work
  └── Cross-genre analysis (nếu data đủ đa dạng)
```

---

## Checklist tổng hợp

### Đợt 1 — Core (MUST HAVE)
- [ ] RUN-01: Random baseline
- [ ] RUN-02: Majority baseline
- [ ] RUN-03: Silence-threshold baseline
- [ ] RUN-B0: Original VAP (HuBERT-EN)
- [ ] RUN-B1: + Wav2Vec2-Vi
- [ ] RUN-B2: + PhoBERT text
- [ ] RUN-B3: + HuTu detector
- [ ] RUN-B4: Full model × 3 seeds
- [ ] ANALYSIS-01: Prosody sensitivity

### Đợt 2 — Ablation (SHOULD HAVE)
- [ ] RUN-04: Text-only baseline
- [ ] RUN-L1: GMU fusion
- [ ] RUN-L2: Bottleneck fusion
- [ ] RUN-L3: 2-stage training
- [ ] RUN-L4: WavLM encoder
- [ ] ANALYSIS-02: Gap duration breakdown
- [ ] ANALYSIS-04: Vietnamese marker deep-dive
- [ ] ANALYSIS-05: Qualitative examples
- [ ] Bootstrap CI cho tất cả metrics
- [ ] Permutation test cho tất cả cặp so sánh

### Đợt 3 — Bonus (NICE TO HAVE)
- [ ] ANALYSIS-03: Data efficiency / learning curve
- [ ] VALIDATION-01: DER check (manual)
- [ ] VALIDATION-02: WER check (manual)
- [ ] VALIDATION-03: Event annotation check
- [ ] Cross-genre generalization

---

## Cần implement trước khi chạy

| File cần tạo | Cho run nào | Ưu tiên |
|---------------|-------------|---------|
| `configs/config_b0_original.yaml` | B0 | P0 |
| `configs/config_b1_vi_encoder.yaml` | B1 | P0 |
| `configs/config_b2_add_text.yaml` | B2 | P0 |
| `configs/config_b3_add_hutu.yaml` | B3 | P0 |
| `configs/config_text_only.yaml` | RUN-04 | P1 |
| `configs/config_fusion_gmu.yaml` | L1 | P1 |
| `configs/config_fusion_bottleneck.yaml` | L2 | P1 |
| `configs/config_2stage.yaml` | L3 | P1 |
| `configs/config_encoder_wavlm.yaml` | L4 | P1 |
| `scripts/silence_threshold_baseline.py` | RUN-03 | P0 |
| `scripts/flatten_pitch.py` | ANALYSIS-01 | P1 |
| Thêm `--seed` vào `train.py` | B4 × 3 seeds | P0 |
| Thêm `enabled: false` support cho linguistic encoder | B0, B1 | P0 |

---

## Ghi chú cho giảng viên

1. **Incremental ablation (B0→B4)** là cấu trúc chính, giống cách các paper YOLOv8/DETR trình bày.
   Mỗi dòng thêm đúng 1 đóng góp → reviewer thấy rõ giá trị từng thành phần.

2. **3 seeds cho full model** đảm bảo kết quả không do may mắn.
   Báo cáo: mean ± std cho mỗi metric.

3. **Prosody sensitivity** là bắt buộc vì tiếng Việt là tonal language.
   Nếu thiếu, reviewer sẽ yêu cầu trong rebuttal → mất thời gian.

4. **Silence-threshold baseline** là standard trong mọi paper VAP.
   Thiếu = reject ngay vì không có fair comparison với rule-based approach.

5. **Data efficiency** (đợt 3) rất mạnh cho narrative "low-resource Vietnamese"
   nhưng tốn GPU → chạy nếu còn thời gian.

6. **Manual validation** (đợt 3) không cần GPU nhưng cần thời gian.
   Nên làm song song với đợt 2.
