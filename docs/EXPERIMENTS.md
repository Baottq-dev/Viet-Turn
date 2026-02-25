# MM-VAP-VI: Kế hoạch thí nghiệm cho paper

> Chi tiết các thí nghiệm cần thực hiện để có đủ kết quả cho paper.
> Target venues: INTERSPEECH, ACL Findings, SIGdial Workshop.

---

## 1. Baselines

**Mục đích:** So sánh MM-VAP-VI với các phương pháp khác để chứng minh tính ưu việt.

### 1.1 Random Baseline

Dự đoán ngẫu nhiên từ 256 classes.

```python
import numpy as np

def random_baseline(labels, num_classes=256, seed=42):
    rng = np.random.RandomState(seed)
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    random_preds = rng.randint(0, num_classes, size=len(valid_labels))
    acc = (random_preds == valid_labels).mean()
    return {"acc": acc}  # Kỳ vọng ~0.4%
```

### 1.2 Majority Class Baseline

Luôn dự đoán class phổ biến nhất (thường là silence class 0).

```python
from collections import Counter

def majority_baseline(train_labels, test_labels):
    valid_train = train_labels[train_labels >= 0]
    majority_class = Counter(valid_train.tolist()).most_common(1)[0][0]
    valid_test = test_labels[test_labels >= 0]
    acc = (valid_test == majority_class).mean()
    return {"acc": acc, "majority_class": majority_class}
```

**Kết quả kỳ vọng:** ~30-50% (phụ thuộc vào tỷ lệ silence trong data).

### 1.3 Audio-only VAP (Stage 1)

Chỉ dùng acoustic encoder (Wav2Vec2) + Transformer, không có text.
Đây chính là kết quả Stage 1 của training.

```bash
python evaluate.py \
    --checkpoint outputs/mm_vap/checkpoint_s1_e10.pt \
    --test-manifest data/vap_manifest_test.json \
    --use-text false
```

**Kết quả hiện tại:** val_acc = 71.2% (Stage 1, Epoch 10).

### 1.4 Text-only Baseline

Chỉ dùng PhoBERT text features, không có audio. Cho thấy text có hữu ích đến mức nào khi dùng độc lập.

```python
class TextOnlyVAP(nn.Module):
    def __init__(self, linguistic_encoder, transformer, projection_head):
        super().__init__()
        self.linguistic_encoder = linguistic_encoder
        self.transformer = transformer
        self.projection_head = projection_head

    def forward(self, texts, num_frames):
        ling = self.linguistic_encoder(texts)  # (B, dim)
        ling_expanded = ling.unsqueeze(1).expand(-1, num_frames, -1)
        contextualized, _ = self.transformer(ling_expanded)
        logits = self.projection_head(contextualized)
        return logits
```

**Kết quả kỳ vọng:** ~40-55% accuracy.

### 1.5 Original VAP (Ekstedt & Skantze, 2022)

Baseline audio-only VAP gốc (tiếng Anh/Thụy Điển), fine-tune trên data tiếng Việt.
Chứng minh rằng architecture multimodal + HuTu vượt trội so với VAP gốc.

```python
class OriginalVAP(nn.Module):
    """Simplified VAP following Ekstedt & Skantze (2022)."""
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960")
        self.transformer = CausalTransformer(num_layers=4, dim=256, ...)
        self.proj = VAPProjectionHead(input_dim=256, num_classes=256)
```

**Ý nghĩa:** Cho thấy Wav2Vec2-Vi tốt hơn HuBERT-English cho tiếng Việt, và multimodal tốt hơn audio-only.

### 1.6 Bảng tổng hợp Baselines

| Model | Audio | Text | HuTu | Acc kỳ vọng | Mục đích |
|-------|-------|------|------|-------------|----------|
| Random | - | - | - | ~0.4% | Lower bound |
| Majority | - | - | - | ~35% | Naive baseline |
| Audio-only (Stage 1) | Wav2Vec2-Vi | - | - | ~71% | Đóng góp audio |
| Text-only | - | PhoBERT-v2 | Yes | ~45% | Đóng góp text |
| Original VAP | HuBERT-EN | - | - | ~60% | Cross-lingual |
| **MM-VAP-VI (Full)** | **Wav2Vec2-Vi** | **PhoBERT-v2** | **Yes** | **~75%** | **Đề xuất** |

> **Lưu ý:** VAQI sử dụng thang điểm 0-100 (ví dụ: 71.0, không phải 0.71).

---

## 2. Ablation Studies

**Mục đích:** Phân tích đóng góp của từng thành phần trong model.

### 2.1 Ma trận Ablation

| Exp | Acoustic | PhoBERT | HuTu | Fusion | Mô tả |
|-----|----------|---------|------|--------|-------|
| A1 | Wav2Vec2-Vi | - | - | - | Audio-only (= Stage 1) |
| A2 | Wav2Vec2-Vi | PhoBERT-v2 | - | Cross-Attn | Không có HuTu |
| A3 | Wav2Vec2-Vi | PhoBERT-v2 | Yes | Cross-Attn | **Full model** |
| A4 | Wav2Vec2-Vi | PhoBERT-v2 | Yes | GMU | So sánh fusion |
| A5 | Wav2Vec2-Vi | PhoBERT-v2 | Yes | Bottleneck | So sánh fusion |
| A6 | WavLM-base | PhoBERT-v2 | Yes | Cross-Attn | So sánh encoder |

### 2.2 Cách chạy từng ablation

#### A2: Không có HuTu

```yaml
# configs/ablation_no_hutu.yaml
linguistic_encoder:
  hutu_detector:
    enabled: false
```

```bash
python train.py --config configs/ablation_no_hutu.yaml \
    --output-dir outputs/ablation_no_hutu
```

#### A4: GMU Fusion

```yaml
fusion:
  type: "gmu"
  dim: 256
  dropout: 0.1
```

```bash
python train.py --config configs/ablation_gmu.yaml \
    --output-dir outputs/ablation_gmu
```

#### A5: Bottleneck Fusion

```yaml
fusion:
  type: "bottleneck"
  dim: 256
  num_heads: 4
  num_latents: 16
  dropout: 0.1
```

#### A6: WavLM encoder

```yaml
acoustic_encoder:
  type: "wavlm"
  pretrained: "microsoft/wavlm-base"
```

### 2.3 Ablation chiến lược Training

**Mục đích:** Chứng minh 3-stage curriculum tốt hơn 1-stage/2-stage.
Reviewer **sẽ hỏi** tại sao chọn 3-stage mà không so sánh với các chiến lược khác.

| Exp | Chiến lược | Mô tả |
|-----|-----------|-------|
| T1 | 1-stage | End-to-end, tất cả module unfreeze từ đầu, 50 epochs |
| T2 | 2-stage | Audio-only (15 ep) → Full (35 ep) |
| T3 | 3-stage | Audio-only (10 ep) → Multimodal (20 ep) → Fine-tune (20 ep) |

#### T1: 1-stage end-to-end

```yaml
training:
  stage1:
    epochs: 50
    freeze: []
    lr:
      acoustic_encoder: 5.0e-5
      linguistic_encoder: 2.0e-5
      fusion: 1.0e-4
      transformer: 5.0e-5
      projection_head: 5.0e-5
  stage2:
    epochs: 0   # Bỏ qua
  stage3:
    epochs: 0   # Bỏ qua
```

```bash
python train.py --config configs/config_1stage.yaml \
    --output-dir outputs/strategy_1stage
```

#### T2: 2-stage

```yaml
training:
  stage1:
    epochs: 15
    freeze: ["linguistic_encoder", "fusion"]
    lr:
      acoustic_encoder: 1.0e-4
      transformer: 1.0e-4
      projection_head: 1.0e-4
  stage2:
    epochs: 35
    freeze: []
    lr:
      acoustic_encoder: 2.0e-5
      linguistic_encoder: 1.0e-5
      fusion: 5.0e-5
      transformer: 5.0e-5
      projection_head: 5.0e-5
  stage3:
    epochs: 0   # Bỏ qua
```

```bash
python train.py --config configs/config_2stage.yaml \
    --output-dir outputs/strategy_2stage
```

#### Bảng kết quả

| Chiến lược | Frame Acc (%) | Shift/Hold BA | VAQI | Hội tụ (epochs) |
|-----------|---------------|---------------|------|-----------------|
| 1-stage end-to-end | | | | |
| 2-stage (audio → full) | | | | |
| **3-stage (đề xuất)** | | | | |

> **Lý do kỳ vọng 3-stage tốt hơn:**
> - Stage 1: acoustic encoder học vững trước khi bị nhiễu bởi text gradients
> - Stage 2: fusion học từ acoustic features đã tốt, không phải từ random
> - Stage 3: fine-tune toàn bộ với LR nhỏ để polish

> **Lưu ý:** `trainer.py` đã hỗ trợ skip stage khi `epochs: 0` (xem Mục 6.3).

### 2.4 Kết quả cần báo cáo

| Metric | A1 | A2 | A3 | A4 | A5 | A6 |
|--------|----|----|----|----|----|----|
| Frame Acc (%) | | | | | | |
| Frame F1 (weighted) | | | | | | |
| Shift/Hold BA | | | | | | |
| BC F1 | | | | | | |
| EoT Latency (ms) | | | | | | |
| VAQI | | | | | | |
| Params (M) | | | | | | |
| Thời gian train (h) | | | | | | |

### 2.5 Phân tích chuyên sâu HuTu Detector

Đây là contribution chính của paper, cần phân tích kỹ:

```python
from src.evaluation.vietnamese_analysis import analyze_marker_impact

# Kết quả kỳ vọng:
# - Frames có yield markers (à, nhé, nhỉ): +5-10% accuracy
# - Frames có hold markers (mà, là, thì): +3-5% accuracy
# - Frames có backchannel markers (ừ, ờ): +8-12% accuracy
```

| Loại Marker | Ví dụ | Số lượng | Acc có HuTu | Acc không HuTu | Delta |
|-------------|-------|---------|-------------|----------------|-------|
| Yield (SFP) | à, nhé, nhỉ | ? | ? | ? | ? |
| Hold (liên từ) | mà, là, thì | ? | ? | ? | ? |
| Backchannel | ừ, ờ, vâng | ? | ? | ? | ? |
| Xin lượt | này, ơi | ? | ? | ? | ? |
| Không có marker | - | ? | ? | ? | ? |

---

## 3. Event-Level Metrics (Tier 2)

Code đã implement trong `src/evaluation/event_metrics.py`.

### 3.1 Các loại sự kiện

| Sự kiện | Định nghĩa | Ground Truth Source |
|---------|-----------|-------------------|
| **SHIFT** | Speaker A dừng, Speaker B bắt đầu nói | VA matrix: S_A 1→0, S_B 0→1 |
| **HOLD** | Speaker A dừng (im lặng), rồi tiếp tục nói | VA matrix: S_A 1→0→1, S_B = 0 |
| **BACKCHANNEL** | Speaker B nói ngắn (≤500ms) trong khi A vẫn giữ sàn | VA matrix: S_B = 1 ngắn (≤25 frames), S_A active trong context ±10 frames |

### 3.2 Cách tính

```python
from src.evaluation.event_metrics import classify_events, compute_event_metrics

pred_probs = torch.softmax(logits, dim=-1)  # (T, 256)

event_data = classify_events(
    probs=pred_probs,
    va_matrix=va_matrix,
    shift_threshold=0.5,
    bc_max_frames=25,       # Tối đa 500ms cho backchannel
    labels=vap_labels,
)

gt_events = event_data["gt_events"]       # (N,) int: 0=hold, 1=shift, 2=bc
pred_events = event_data["pred_events"]   # (N,) int
p_shift = event_data["p_shift"]           # (N,) float: P(shift)
gt_shift = event_data["gt_shift"]         # (N,) float: binary shift labels

metrics = compute_event_metrics(gt_events, pred_events, p_shift, gt_shift)
# → shift_hold_ba, bc_f1, predict_shift_auc, event_3class_ba
```

### 3.3 Metrics cần báo cáo

| Metric | Mô tả | Target |
|--------|--------|--------|
| **Shift/Hold BA** | Balanced Accuracy phân biệt shift vs hold | > 0.65 |
| **BC F1** | F1-score phát hiện backchannel | > 0.50 |
| **Predict Shift AUC** | ROC-AUC dự đoán shift trước khi xảy ra | > 0.70 |
| **3-class BA** | Balanced Accuracy 3 class (shift/hold/BC) | > 0.55 |

### 3.4 Confusion Matrix

```
                  Predicted
              SHIFT  HOLD   BC
GT  SHIFT   [  TP  |  FN  | FN ]
    HOLD    [  FP  |  TP  | FP ]
    BC      [  FP  |  FP  | TP ]
```

Cần báo cáo confusion matrix cho mỗi model/ablation.

---

## 4. Latency Metrics (Tier 3)

Code đã implement trong `src/evaluation/latency_metrics.py`.

### 4.1 End-of-Turn (EoT) Latency

Đo thời gian model cần để nhận ra speaker đã kết thúc lượt nói.

```python
from src.evaluation.latency_metrics import compute_eot_latency

latency = compute_eot_latency(
    p_shift=p_shift_probs,
    gt_shift_regions=gt_shifts,
    threshold=0.5,
    frame_hz=50,
)
# → eot_latency_mean_ms, eot_latency_median_ms, eot_latency_p95_ms, detection_rate
```

**Benchmarks (từ literature):**

| Metric | Con người | Model tốt | Chấp nhận được |
|--------|-----------|-----------|----------------|
| Median Latency | ~200ms | < 300ms | < 500ms |
| P95 Latency | ~500ms | < 800ms | < 1500ms |
| Detection Rate | 100% | > 90% | > 80% |

### 4.2 False Positive Rate theo ngưỡng Latency

Khi model phát hiện shift nhanh, nó có thể bị false positive (ngắt lời).
Metric này đo trade-off giữa tốc độ và độ chính xác.

```python
from src.evaluation.latency_metrics import compute_fpr_at_thresholds

fpr = compute_fpr_at_thresholds(
    p_shift=p_shift_probs,
    gt_shift=gt_shift_binary,
    thresholds_ms=[100, 200, 300, 500, 1000],
    frame_hz=50,
)
```

| Ngưỡng Latency | FPR (Audio-only) | FPR (MM-VAP-VI) | Delta |
|-----------------|------------------|-----------------|-------|
| 100ms | ? | ? | ? |
| 200ms | ? | ? | ? |
| 300ms | ? | ? | ? |
| 500ms | ? | ? | ? |
| 1000ms | ? | ? | ? |

### 4.3 VAQI — Voice Agent Quality Index

Metric tổng hợp đánh giá chất lượng VAP cho ứng dụng voice agent thực tế.

```python
from src.evaluation.latency_metrics import compute_vaqi

vaqi = compute_vaqi(
    p_shift=p_shift_probs,
    gt_shift_regions=gt_shifts,
    gt_hold_regions=gt_holds,
    threshold=0.5,
    frame_hz=50,
    max_latency_ms=2000.0,
)
# → vaqi (0-100), vaqi_interruption_rate, vaqi_missed_response_rate,
#   vaqi_latency_score, vaqi_median_latency_ms
```

**Các thành phần VAQI:**

| Thành phần | Mô tả | Trọng số | Target |
|------------|--------|----------|--------|
| Interruption Rate (I) | Tỷ lệ model ngắt lời (dự đoán shift khi hold) | 0.4 | < 10% |
| Missed Response (M) | Tỷ lệ model bỏ lỡ shift | 0.4 | < 15% |
| Latency Score (L) | Log-scaled median latency, chuẩn hóa về [0,1] | 0.2 | < 0.3 |
| **VAQI Total** | **100 × (1 − [0.4×I + 0.4×M + 0.2×L])** | **0-100** | **> 65** |

### 4.4 EoT Levenshtein (Sequence-based)

Đánh giá độ chính xác của chuỗi EoT predictions so với ground truth.

```python
from src.evaluation.latency_metrics import compute_eot_levenshtein

lev = compute_eot_levenshtein(
    gt_eot_frames=gt_eot_list,
    pred_eot_frames=pred_eot_list,
    tolerance_frames=25,    # 500ms tolerance
    frame_hz=50,
)
# → eot_precision, eot_recall, eot_f1, eot_mean_position_error_ms
```

---

## 5. Kiểm định thống kê

### 5.1 Paired Permutation Test

**Mục đích:** Chứng minh sự khác biệt giữa 2 models là **có ý nghĩa thống kê**, không phải do ngẫu nhiên.

```python
from src.evaluation.statistical import permutation_test

result = permutation_test(
    scores_a=audio_only_scores,   # Model A (baseline)
    scores_b=mmvap_scores,        # Model B (đề xuất)
    n_permutations=10000,
    seed=42,
)
# → observed_diff, p_value, significant_at_05, significant_at_01
```

> **Lưu ý quan trọng:**
> - Cần **ít nhất 20-30 conversations** để permutation test có power tốt
> - Hiện tại chỉ có 5 test files → **không đủ** cho statistical significance
> - Đây là lý do chính cần tăng data lên 30-50+ conversations

### 5.2 Bootstrap Confidence Intervals

Tính khoảng tin cậy 95% cho mỗi metric bằng bootstrap.

```python
evaluator = MMVAPEvaluator(model, test_loader, device="cuda")
results = evaluator.evaluate_with_bootstrap(
    n_bootstrap=1000,
    ci=0.95,
    use_text=True,
)
# Mỗi metric sẽ có: mean, ci_lower, ci_upper
```

**Cách báo cáo trong paper:**

```
Frame Accuracy: 75.2% (95% CI: 73.1–77.3)
Shift/Hold BA:  0.68  (95% CI: 0.64–0.72)
BC F1:          0.54  (95% CI: 0.48–0.60)
EoT Latency:   285ms (95% CI: 240–330)
VAQI:           71.0  (95% CI: 66.0–76.0)
```

### 5.3 Kiểm định cho từng cặp so sánh

| So sánh | Giả thuyết | Test |
|---------|-----------|------|
| Full vs Audio-only | Text cải thiện accuracy | Permutation, p < 0.05 |
| Full vs Không HuTu | HuTu cải thiện accuracy | Permutation, p < 0.05 |
| Full vs Original VAP | Adaptation tiếng Việt hiệu quả | Permutation, p < 0.05 |
| Cross-Attn vs GMU | Loại fusion có ảnh hưởng | Permutation, p < 0.05 |

### 5.4 Effect Size (Cohen's d)

```python
def cohens_d(scores_a, scores_b):
    import numpy as np
    na, nb = len(scores_a), len(scores_b)
    var_a, var_b = np.var(scores_a, ddof=1), np.var(scores_b, ddof=1)
    pooled_std = np.sqrt(((na-1)*var_a + (nb-1)*var_b) / (na+nb-2))
    return (np.mean(scores_b) - np.mean(scores_a)) / pooled_std

# |d| < 0.2  : không đáng kể
# 0.2 ≤ |d| < 0.5 : nhỏ
# 0.5 ≤ |d| < 0.8 : trung bình
# |d| ≥ 0.8  : lớn
```

---

## 6. Evaluation Script

> **Trạng thái:** `evaluate.py` **chưa implement**. Hiện tại evaluation logic nằm trong
> `src/evaluation/evaluator.py` (class `MMVAPEvaluator`). Cần tạo file `evaluate.py`
> ở root project trước khi chạy các lệnh bên dưới.

### 6.1 Cách chạy

```bash
# Full evaluation với bootstrap CI
python evaluate.py \
    --checkpoint outputs/mm_vap/best_model.pt \
    --test-manifest data/vap_manifest_test.json \
    --config configs/config.yaml \
    --bootstrap 1000 \
    --output outputs/eval_results.json

# So sánh 2 models (permutation test)
python evaluate.py \
    --checkpoint-a outputs/mm_vap/best_model.pt \
    --checkpoint-b outputs/ablation_no_hutu/best_model.pt \
    --test-manifest data/vap_manifest_test.json \
    --compare \
    --output outputs/comparison_results.json
```

### 6.2 Định dạng output

```json
{
  "model": "MM-VAP-VI (Full)",
  "test_files": 50,
  "tier1_frame": {
    "accuracy": {"mean": 0.752, "ci_lower": 0.731, "ci_upper": 0.773},
    "weighted_f1": {"mean": 0.718, "ci_lower": 0.695, "ci_upper": 0.741},
    "perplexity": {"mean": 12.3, "ci_lower": 11.1, "ci_upper": 13.5},
    "ece": {"mean": 0.045, "ci_lower": 0.038, "ci_upper": 0.052}
  },
  "tier2_event": {
    "shift_hold_ba": {"mean": 0.68, "ci_lower": 0.64, "ci_upper": 0.72},
    "bc_f1": {"mean": 0.54, "ci_lower": 0.48, "ci_upper": 0.60},
    "predict_shift_auc": {"mean": 0.74, "ci_lower": 0.70, "ci_upper": 0.78}
  },
  "tier3_latency": {
    "eot_latency_median_ms": {"mean": 285, "ci_lower": 240, "ci_upper": 330},
    "vaqi": {"mean": 71.0, "ci_lower": 66.0, "ci_upper": 76.0}
  },
  "vietnamese_analysis": {
    "yield_marker_accuracy_delta": 0.08,
    "hold_marker_accuracy_delta": 0.04,
    "backchannel_marker_accuracy_delta": 0.11
  }
}
```

### 6.3 trainer.py skip stage khi epochs = 0

`VAPTrainer.train_stage()` đã có guard để skip stage khi `epochs: 0`:

```python
def train_stage(self, stage: int, start_epoch: int = 1):
    stage_cfg = self.config["training"][f"stage{stage}"]
    num_epochs = stage_cfg["epochs"]
    if num_epochs == 0:
        print(f"\n  Stage {stage}: skipped (epochs=0)")
        return
    # ... phần còn lại
```

**Đã fix** → có thể chạy config 1-stage và 2-stage bình thường.

---

## 7. Mẫu bảng cho paper

### Table 1: Kết quả chính

```
+------------------+----------+-------+----------+-------+--------+------+
| Model            | Frame    | F1    | Shift/   | BC    | EoT    | VAQI |
|                  | Acc (%)  | (w)   | Hold BA  | F1    | Lat(ms)|      |
+------------------+----------+-------+----------+-------+--------+------+
| Random           |  0.4     | 0.002 |  0.50    | 0.00  |   -    |   -  |
| Majority         | 35.1     | 0.18  |  0.50    | 0.00  |   -    |   -  |
| Original VAP     | 62.3     | 0.58  |  0.58    | 0.32  | 450    | 52.0 |
| Audio-only (S1)  | 71.2     | 0.67  |  0.63    | 0.42  | 340    | 61.0 |
| + Text (no HuTu) | 73.5     | 0.70  |  0.65    | 0.48  | 310    | 66.0 |
| + HuTu (Full)    | 75.2*    | 0.72* |  0.68*   | 0.54* | 285*   | 71.0*|
+------------------+----------+-------+----------+-------+--------+------+
  * p < 0.05 vs Audio-only (paired permutation test, n=50 conversations)
```

### Table 2: Ablation Study

```
+------------------+----------+----------+-------+
| Variant          | Frame    | Shift/   | VAQI  |
|                  | Acc (%)  | Hold BA  |       |
+------------------+----------+----------+-------+
| Full model       | 75.2     |  0.68    | 71.0  |
|  − HuTu          | 73.5     |  0.65    | 66.0  |
|  − PhoBERT       | 71.2     |  0.63    | 61.0  |
|  − Cross-Attn    | 74.1     |  0.66    | 68.0  |
|   (dùng GMU)     |          |          |       |
|  − Wav2Vec2-Vi   | 68.3     |  0.59    | 55.0  |
|   (dùng WavLM-EN)|          |          |       |
+------------------+----------+----------+-------+
```

### Table 3: Tác động HuTu Marker

```
+------------------+--------+---------+-----------+-------+
| Loại Marker      | Số lg  | Acc     | Acc       | Delta |
|                  |        | có HuTu | không HuTu|       |
+------------------+--------+---------+-----------+-------+
| Yield (SFP)      | 1,234  | 82.1%   | 74.3%     | +7.8  |
| Hold (liên từ)   | 2,456  | 78.5%   | 74.1%     | +4.4  |
| Backchannel      |   567  | 71.2%   | 59.8%     | +11.4 |
| Xin lượt         |   234  | 69.3%   | 63.2%     | +6.1  |
| Không marker     | 15,678 | 74.8%   | 73.9%     | +0.9  |
+------------------+--------+---------+-----------+-------+
```

### Ý tưởng hình vẽ

1. **Training curves** — Loss/Acc qua 3 stages (đã có trong `training_history.json`)
2. **Latency CDF** — Phân phối tích lũy EoT latency
3. **FPR-Latency tradeoff** — FPR tại các ngưỡng latency khác nhau
4. **Confusion matrix** — 3-class (shift/hold/BC)
5. **Qualitative examples** — 2-3 đoạn hội thoại minh họa predictions
6. **HuTu bar chart** — Cải thiện accuracy theo từng loại marker

---

## 8. Yêu cầu dữ liệu

### 8.1 Hiện trạng

| Metric | Hiện tại | Cần cho paper |
|--------|---------|--------------|
| Tổng audio | ~5.5h (10 videos) | 20-50h (40-100 videos) |
| Train files | 33 segments | 150+ segments |
| Val files | 4 segments | 20+ segments |
| Test files | 5 segments | 30+ segments |
| Tổng frames | 1M | 4-10M |

### 8.2 Cách tăng dữ liệu

1. Tìm thêm YouTube videos (podcast/phỏng vấn 2 người tiếng Việt):

```bash
# Thêm URLs vào scripts/urls.txt, rồi chạy lại pipeline
python scripts/00_download_audio.py --output data/audio
python scripts/00b_split_audio.py --input data/audio --output data/audio_split
# ... (pipeline tự động)
```

2. Cần đảm bảo **đa dạng**:
   - Phương ngữ Bắc / Trung / Nam
   - Giới tính nam / nữ
   - Hội thoại formal / informal
   - Đa dạng chủ đề

3. Annotate thông tin phương ngữ cho mỗi conversation (để dùng `analyze_per_dialect`).

### 8.3 Chiến lược chia dữ liệu

- Split theo **conversation** (không phải theo window) để tránh data leakage
- Giữ tỷ lệ 80/10/10 (train/val/test)
- Đảm bảo mỗi phương ngữ có mặt trong cả 3 splits

---

## 9. Checklist trước khi nộp paper

### Dữ liệu
- [ ] 30+ giờ audio (40+ videos)
- [ ] 3 phương ngữ (Bắc/Trung/Nam) có mặt
- [ ] Test set ≥ 30 conversations
- [ ] Annotation phương ngữ cho mỗi file

### Thí nghiệm
- [ ] Full model đã train (3 stages)
- [ ] Random baseline
- [ ] Majority baseline
- [ ] Audio-only baseline (Stage 1)
- [ ] Text-only baseline
- [ ] Original VAP baseline (tùy chọn nhưng nên có)
- [ ] Ablation không HuTu
- [ ] Ablation loại fusion (GMU, Bottleneck)
- [ ] Ablation encoder (WavLM vs Wav2Vec2-Vi)
- [ ] **Ablation chiến lược training (1-stage vs 2-stage vs 3-stage)**

### Metrics
- [ ] Tier 1: Frame Acc, F1, Perplexity, ECE
- [ ] Tier 2: Shift/Hold BA, BC F1, Shift AUC
- [ ] Tier 3: EoT Latency, đường cong FPR, VAQI
- [ ] Phân tích tiếng Việt: Accuracy theo từng marker
- [ ] Phân tích theo phương ngữ

### Thống kê
- [ ] Bootstrap 95% CI cho mỗi metric
- [ ] Permutation test (p < 0.05) cho mỗi cặp so sánh
- [ ] Cohen's d effect sizes
- [ ] ≥ 30 test conversations để đảm bảo power

### Hình vẽ
- [ ] Training curves (3 stages)
- [ ] Latency CDF
- [ ] FPR-Latency tradeoff
- [ ] Confusion matrix
- [ ] HuTu marker impact bar chart
- [ ] Qualitative examples

### Viết paper
- [ ] Abstract
- [ ] Introduction + motivation (khoảng trống turn-taking tiếng Việt)
- [ ] Related work (VAP, Vietnamese NLP, multimodal)
- [ ] Method (architecture, HuTu, 3-stage training)
- [ ] Experiments (data, baselines, ablations)
- [ ] Results + Analysis
- [ ] Conclusion

---

## 10. Timeline đề xuất

| Tuần | Công việc |
|------|----------|
| 1–2 | Thu thập thêm dữ liệu (30+ videos), chạy pipeline |
| 3 | Train full model + baselines + ablations |
| 4 | Chạy evaluation, kiểm định thống kê |
| 5–6 | Viết paper |
| 7 | Review nội bộ, chỉnh sửa |
| 8 | Nộp paper |
