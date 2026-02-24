# MM-VAP-VI: Experiment Plan for Paper Submission

> Chi tiet cac thi nghiem can thuc hien de co du ket qua cho paper.
> Target venues: INTERSPEECH, ACL Findings, SIGdial Workshop.

---

## 1. Baselines

Muc dich: So sanh MM-VAP-VI voi cac phuong phap khac de chung minh tinh uu viet.

### 1.1 Random Baseline

**Mo ta:** Du doan ngau nhien tu 256 classes.

**Cach chay:**

```python
# Trong evaluate script, them:
import numpy as np

def random_baseline(labels, num_classes=256, seed=42):
    """Random prediction baseline."""
    rng = np.random.RandomState(seed)
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    random_preds = rng.randint(0, num_classes, size=len(valid_labels))
    acc = (random_preds == valid_labels).mean()
    return {"acc": acc}  # Expected ~0.4%
```

**Ket qua ky vong:** ~0.4% accuracy (1/256).

### 1.2 Majority Class Baseline

**Mo ta:** Luon du doan class pho bien nhat (thuong la silence class 0).

**Cach chay:**

```python
from collections import Counter

def majority_baseline(train_labels, test_labels):
    """Always predict most common class."""
    valid_train = train_labels[train_labels >= 0]
    majority_class = Counter(valid_train.tolist()).most_common(1)[0][0]
    valid_test = test_labels[test_labels >= 0]
    acc = (valid_test == majority_class).mean()
    return {"acc": acc, "majority_class": majority_class}
```

**Ket qua ky vong:** ~30-50% (phu thuoc vao ty le silence trong data).

### 1.3 Audio-only VAP (Stage 1 model)

**Mo ta:** Chi dung acoustic encoder (Wav2Vec2) + Transformer, khong co text.
Day chinh la ket qua Stage 1 cua training.

**Cach chay:**

```bash
# Su dung model sau Stage 1 (da co tu training)
# Checkpoint: outputs/mm_vap/checkpoint_s1_e10.pt

python evaluate.py \
    --checkpoint outputs/mm_vap/checkpoint_s1_e10.pt \
    --test-manifest data/vap_manifest_test.json \
    --use-text false
```

**Ket qua hien tai:** val_acc = 71.2% (tu Stage 1 Epoch 10).

### 1.4 Text-only Baseline

**Mo ta:** Chi dung PhoBERT text features, khong co audio. Baseline nay cho thay
text co huu ich den muc nao khi dung doc lap.

**Cach implement:**

```python
# Tao 1 model variant chi co linguistic encoder + transformer + projection
# Khong can acoustic encoder

class TextOnlyVAP(nn.Module):
    def __init__(self, linguistic_encoder, transformer, projection_head):
        super().__init__()
        self.linguistic_encoder = linguistic_encoder
        self.transformer = transformer
        self.projection_head = projection_head

    def forward(self, texts, num_frames):
        # Linguistic encoding -> expand to all frames
        ling = self.linguistic_encoder(texts)  # (B, dim)
        ling_expanded = ling.unsqueeze(1).expand(-1, num_frames, -1)  # (B, T, dim)
        contextualized, _ = self.transformer(ling_expanded)
        logits = self.projection_head(contextualized)
        return logits
```

**Ket qua ky vong:** ~40-55% accuracy (text giup du doan pattern nhung khong co
timing chinh xac tu audio).

### 1.5 Original VAP (Ekstedt & Skantze, 2022)

**Mo ta:** Baseline audio-only VAP goc (tieng Anh/Thuy Dien), fine-tune tren data
tieng Viet. Chung minh rang architecture cua chung ta (multimodal + HuTu)
vuot troi so voi VAP goc.

**Cach implement:**

```python
# Dung CPC/HuBERT encoder thay vi Wav2Vec2-Vi
# Khong co text branch, khong co HuTu detector
# Architecture: CPC -> Transformer -> Projection

# Option 1: Dung pretrained VAP model, fine-tune tren Vietnamese data
# Option 2: Train tu dau voi HuBERT-base + Transformer

class OriginalVAP(nn.Module):
    """Simplified VAP following Ekstedt & Skantze (2022)."""
    def __init__(self):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960")
        self.transformer = CausalTransformer(num_layers=4, dim=256, ...)
        self.proj = VAPProjectionHead(input_dim=256, num_classes=256)
```

**Y nghia:** Cho thay Wav2Vec2-Vi tot hon HuBERT-English cho tieng Viet,
va multimodal tot hon audio-only.

### 1.6 Bang tong hop Baselines

| Model | Audio | Text | HuTu | Expected Acc | Purpose |
|-------|-------|------|------|-------------|---------|
| Random | - | - | - | ~0.4% | Lower bound |
| Majority | - | - | - | ~35% | Naive baseline |
| Audio-only (Stage 1) | Wav2Vec2-Vi | - | - | ~71% | Audio contribution |
| Text-only | - | PhoBERT-v2 | Yes | ~45% | Text contribution |
| Original VAP | HuBERT-EN | - | - | ~60% | Cross-lingual |
| **MM-VAP-VI (Full)** | **Wav2Vec2-Vi** | **PhoBERT-v2** | **Yes** | **~75%** | **Proposed** |

> **Luu y:** VAQI su dung thang diem 0-100 (vi du: 71.0, khong phai 0.71).

---

## 2. Ablation Studies

Muc dich: Phan tich dong gop cua tung thanh phan trong model.

### 2.1 Ablation Matrix

| Exp | Acoustic | PhoBERT | HuTu | Fusion | Description |
|-----|----------|---------|------|--------|-------------|
| A1 | Wav2Vec2-Vi | - | - | - | Audio-only (= Stage 1) |
| A2 | Wav2Vec2-Vi | PhoBERT-v2 | - | Cross-Attn | No HuTu |
| A3 | Wav2Vec2-Vi | PhoBERT-v2 | Yes | Cross-Attn | **Full model** |
| A4 | Wav2Vec2-Vi | PhoBERT-v2 | Yes | GMU | Fusion comparison |
| A5 | Wav2Vec2-Vi | PhoBERT-v2 | Yes | Bottleneck | Fusion comparison |
| A6 | WavLM-base | PhoBERT-v2 | Yes | Cross-Attn | Encoder comparison |

### 2.2 Cach chay tung ablation

#### A2: No HuTu (tat hutu_detector)

Sua `configs/config.yaml`:

```yaml
linguistic_encoder:
  hutu_detector:
    enabled: false     # <-- Tat HuTu
```

```bash
python train.py --config configs/ablation_no_hutu.yaml \
    --output-dir outputs/ablation_no_hutu
```

#### A4: GMU Fusion

```yaml
fusion:
  type: "gmu"          # <-- Doi tu cross_attention sang gmu
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
  num_latents: 16      # <-- Bottleneck specific
  dropout: 0.1
```

#### A6: WavLM encoder

```yaml
acoustic_encoder:
  type: "wavlm"
  pretrained: "microsoft/wavlm-base"   # <-- Doi encoder
```

### 2.3 Training Strategy Ablation

Muc dich: Chung minh 3-stage curriculum tot hon 1-stage/2-stage.
Reviewer **se hoi** tai sao chon 3-stage ma khong so sanh voi cac chien luoc khac.

| Exp | Strategy | Description |
|-----|---------|-------------|
| T1 | 1-stage | End-to-end, tat ca module unfreeze tu dau, 50 epochs |
| T2 | 2-stage | Audio-only (15 ep) -> Full (35 ep) |
| T3 | 3-stage | Audio-only (10 ep) -> Multimodal (20 ep) -> Fine-tune (20 ep) |

#### T1: 1-stage end-to-end

Tao `configs/config_1stage.yaml` (copy tu config.yaml, thay doi `training`):

```yaml
training:
  stage1:
    epochs: 50
    freeze: []       # Khong freeze gi
    lr:
      acoustic_encoder: 5.0e-5
      linguistic_encoder: 2.0e-5
      fusion: 1.0e-4
      transformer: 5.0e-5
      projection_head: 5.0e-5
  stage2:
    epochs: 0        # Skip
  stage3:
    epochs: 0        # Skip
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
    epochs: 0        # Skip
```

```bash
python train.py --config configs/config_2stage.yaml \
    --output-dir outputs/strategy_2stage
```

#### Bang ket qua

| Strategy | Frame Acc (%) | Shift/Hold BA | VAQI | Convergence (epochs) |
|---------|---------------|---------------|------|---------------------|
| 1-stage end-to-end | | | | |
| 2-stage (audio -> full) | | | | |
| **3-stage (proposed)** | | | | |

> **Ly do ky vong 3-stage tot hon:**
> - Stage 1 cho acoustic encoder hoc vung truoc khi bi nhieu boi text gradients
> - Stage 2 cho fusion hoc tu acoustic features da tot, ko phai tu random
> - Stage 3 fine-tune toan bo voi LR nho de polish

> **Luu y:** Can fix `trainer.py` de skip stage khi `epochs: 0` (xem Section 6.3).

### 2.4 Ket qua can bao cao

Moi ablation can bao cao:

| Metric | A1 | A2 | A3 | A4 | A5 | A6 |
|--------|----|----|----|----|----|----|
| Frame Acc (%) | | | | | | |
| Frame F1 (weighted) | | | | | | |
| Shift/Hold BA | | | | | | |
| BC F1 | | | | | | |
| EoT Latency (ms) | | | | | | |
| VAQI | | | | | | |
| Params (M) | | | | | | |
| Train time (h) | | | | | | |

### 2.5 HuTu Detector Deep Dive

Day la contribution chinh cua paper, can phan tich ky:

```python
# So sanh accuracy tai cac frame co/khong co discourse markers

from src.evaluation.vietnamese_analysis import analyze_marker_impact

# Ket qua ky vong:
# - Frames co yield markers (a, nhe, nhi): +5-10% accuracy
# - Frames co hold markers (ma, la, thi): +3-5% accuracy
# - Frames co backchannel markers (u, o): +8-12% accuracy
```

**Bang can bao cao:**

| Marker Category | Examples | #Occurrences | Acc with HuTu | Acc without HuTu | Delta |
|----------------|----------|-------------|---------------|-----------------|-------|
| Yield (SFP) | a, nhe, nhi | ? | ? | ? | ? |
| Hold (conjunction) | ma, la, thi | ? | ? | ? | ? |
| Backchannel | u, o, vang | ? | ? | ? | ? |
| Turn request | nay, oi | ? | ? | ? | ? |
| No marker | - | ? | ? | ? | ? |

---

## 3. Event-Level Metrics (Tier 2)

Code da implement trong `src/evaluation/event_metrics.py`.

### 3.1 Event Types

Ba loai su kien turn-taking:

| Event | Dinh nghia | Ground Truth Source |
|-------|-----------|-------------------|
| **SHIFT** | Speaker A dung, Speaker B bat dau noi | VA matrix: S_A 1->0, S_B 0->1 |
| **HOLD** | Speaker A dung (im lang), roi tiep tuc noi | VA matrix: S_A 1->0->1, S_B = 0 |
| **BACKCHANNEL** | Speaker B noi ngan (<1s) trong khi A van noi | VA matrix: S_B = 1 ngan, S_A = 1 |

### 3.2 Cach tinh

```python
from src.evaluation.event_metrics import classify_events, compute_event_metrics

# classify_events() tinh ca GT events (tu VA matrix) va predicted events
# (tu model probs) trong cung 1 loi goi:
pred_probs = torch.softmax(logits, dim=-1)  # (T, 256)

event_data = classify_events(
    probs=pred_probs,               # Bat buoc, khong duoc None
    va_matrix=va_matrix,            # Ground truth VA matrix
    shift_threshold=0.5,
    bc_max_frames=50,               # Max 1s for backchannel
    labels=vap_labels,              # Mask invalid frames
)

# Ket qua la dict, can unpack:
gt_events = event_data["gt_events"]       # (N,) int: 0=hold, 1=shift, 2=bc
pred_events = event_data["pred_events"]   # (N,) int
p_shift = event_data["p_shift"]           # (N,) float: P(shift)
gt_shift = event_data["gt_shift"]         # (N,) float: binary shift labels

# Tinh metrics
metrics = compute_event_metrics(gt_events, pred_events, p_shift, gt_shift)
# -> shift_hold_ba, bc_f1, predict_shift_auc, event_3class_ba
```

### 3.3 Metrics can bao cao

| Metric | Mo ta | Target |
|--------|------|--------|
| **Shift/Hold BA** | Balanced Accuracy phan biet shift vs hold | > 0.65 |
| **BC F1** | F1-score phat hien backchannel | > 0.50 |
| **Predict Shift AUC** | ROC-AUC du doan shift truoc khi xay ra | > 0.70 |
| **3-class BA** | Balanced Accuracy 3 class (shift/hold/BC) | > 0.55 |

### 3.4 Confusion Matrix

```
                  Predicted
              SHIFT  HOLD   BC
GT  SHIFT   [  TP  |  FN  | FN ]
    HOLD    [  FP  |  TP  | FP ]
    BC      [  FP  |  FP  | TP ]
```

Can bao cao confusion matrix nay cho moi model/ablation.

---

## 4. Latency Metrics (Tier 3)

Code da implement trong `src/evaluation/latency_metrics.py`.

### 4.1 End-of-Turn (EoT) Latency

Do thoi gian model can de nhan ra speaker da ket thuc luot noi.

```python
from src.evaluation.latency_metrics import compute_eot_latency

latency = compute_eot_latency(
    p_shift=p_shift_probs,          # (T,) xac suat shift
    gt_shift_regions=gt_shifts,      # [(start, end), ...]
    threshold=0.5,
    frame_hz=50,
)
# -> eot_latency_mean_ms, eot_latency_median_ms, eot_latency_p95_ms
#    detection_rate
```

**Target benchmarks (tu literature):**

| Metric | Human | Good Model | Acceptable |
|--------|-------|-----------|-----------|
| Median Latency | ~200ms | < 300ms | < 500ms |
| P95 Latency | ~500ms | < 800ms | < 1500ms |
| Detection Rate | 100% | > 90% | > 80% |

### 4.2 False Positive Rate at Latency Thresholds

Khi model phat hien shift nhanh, no co the bi false positive (ngat loi).
Metric nay do trade-off giua speed va accuracy.

```python
from src.evaluation.latency_metrics import compute_fpr_at_thresholds

fpr = compute_fpr_at_thresholds(
    p_shift=p_shift_probs,
    gt_shift=gt_shift_binary,
    thresholds_ms=[100, 200, 300, 500, 1000],
    frame_hz=50,
)
# -> fpr_at_100ms, fpr_at_200ms, ...
```

**Bang can bao cao:**

| Latency Threshold | FPR (Audio-only) | FPR (MM-VAP-VI) | Delta |
|-------------------|------------------|-----------------|-------|
| 100ms | ? | ? | ? |
| 200ms | ? | ? | ? |
| 300ms | ? | ? | ? |
| 500ms | ? | ? | ? |
| 1000ms | ? | ? | ? |

### 4.3 VAQI - Voice Agent Quality Index

Metric tong hop danh gia chat luong VAP cho ung dung voice agent thuc te.

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
# -> vaqi (0-100), vaqi_interruption_rate, vaqi_missed_response_rate,
#    vaqi_latency_score, vaqi_median_latency_ms
```

**VAQI components:**

| Component | Mo ta | Weight | Target |
|-----------|------|--------|--------|
| Interruption Rate (I) | Ty le model ngat loi (du doan shift khi hold) | 0.4 | < 10% |
| Missed Response (M) | Ty le model bo lo shift | 0.4 | < 15% |
| Latency Score (L) | log-scaled median latency, normalized to [0,1] | 0.2 | < 0.3 |
| **VAQI Total** | **100 × (1 - [0.4×I + 0.4×M + 0.2×L])** | **0-100** | **> 65** |

### 4.4 EoT Levenshtein (Sequence-based)

Danh gia do chinh xac cua chuoi EoT predictions so voi ground truth.

```python
from src.evaluation.latency_metrics import compute_eot_levenshtein

lev = compute_eot_levenshtein(
    gt_eot_frames=gt_eot_list,
    pred_eot_frames=pred_eot_list,
    tolerance_frames=25,          # 500ms tolerance
    frame_hz=50,
)
# -> eot_precision, eot_recall, eot_f1, eot_mean_position_error_ms
```

---

## 5. Statistical Tests

### 5.1 Paired Permutation Test

Muc dich: Chung minh su khac biet giua 2 models la **co y nghia thong ke**,
khong phai do ngau nhien.

```python
from src.evaluation.statistical import permutation_test

# scores_a, scores_b: per-conversation metric scores
# Vi du: shift_hold_ba cua moi conversation

result = permutation_test(
    scores_a=audio_only_scores,     # Model A (baseline)
    scores_b=mmvap_scores,          # Model B (proposed)
    n_permutations=10000,
    seed=42,
)
# -> observed_diff, p_value, significant_at_05, significant_at_01
```

**Luu y quan trong:**
- Can **it nhat 20-30 conversations** de permutation test co power tot
- Hien tai chi co 5 test files -> **khong du** cho statistical significance
- Day la ly do chinh can tang data len 30-50+ conversations

### 5.2 Bootstrap Confidence Intervals

Tinh khoang tin cay 95% cho moi metric bang bootstrap.

```python
# MMVAPEvaluator da co san evaluate_with_bootstrap()

evaluator = MMVAPEvaluator(model, test_loader, device="cuda")
results = evaluator.evaluate_with_bootstrap(
    n_bootstrap=1000,
    ci=0.95,
    use_text=True,
)
# Moi metric se co: mean, ci_lower, ci_upper
```

**Cach bao cao trong paper:**

```
Frame Accuracy: 75.2% (95% CI: 73.1-77.3)
Shift/Hold BA:  0.68 (95% CI: 0.64-0.72)
BC F1:          0.54 (95% CI: 0.48-0.60)
EoT Latency:   285ms (95% CI: 240-330)
VAQI:           71.0/100 (95% CI: 66.0-76.0)
```

### 5.3 Per-comparison Statistical Tests

Moi cap so sanh trong ablation can test rieng:

| Comparison | Hypothesis | Test |
|-----------|-----------|------|
| Full vs Audio-only | Text improves accuracy | Permutation, p < 0.05 |
| Full vs No-HuTu | HuTu improves accuracy | Permutation, p < 0.05 |
| Full vs Original VAP | Vietnamese adaptation helps | Permutation, p < 0.05 |
| Cross-Attn vs GMU | Fusion type matters | Permutation, p < 0.05 |

### 5.4 Effect Size

Ngoai p-value, bao cao Cohen's d:

```python
def cohens_d(scores_a, scores_b):
    """Compute Cohen's d effect size."""
    import numpy as np
    na, nb = len(scores_a), len(scores_b)
    var_a, var_b = np.var(scores_a, ddof=1), np.var(scores_b, ddof=1)
    pooled_std = np.sqrt(((na-1)*var_a + (nb-1)*var_b) / (na+nb-2))
    return (np.mean(scores_b) - np.mean(scores_a)) / pooled_std

# Interpretation:
# |d| < 0.2: negligible
# 0.2 <= |d| < 0.5: small
# 0.5 <= |d| < 0.8: medium
# |d| >= 0.8: large
```

---

## 6. Evaluation Script

> **Status:** `evaluate.py` **chua implement**. Hien tai evaluation logic nam trong
> `src/evaluation/evaluator.py` (class `MMVAPEvaluator`). Can tao file `evaluate.py`
> o root project truoc khi chay cac lenh ben duoi.

### 6.1 Cach chay (sau khi tao evaluate.py)

```bash
# Full evaluation voi bootstrap CI
python evaluate.py \
    --checkpoint outputs/mm_vap/best_model.pt \
    --test-manifest data/vap_manifest_test.json \
    --config configs/config.yaml \
    --bootstrap 1000 \
    --output outputs/eval_results.json

# So sanh 2 models (permutation test)
python evaluate.py \
    --checkpoint-a outputs/mm_vap/best_model.pt \
    --checkpoint-b outputs/ablation_no_hutu/best_model.pt \
    --test-manifest data/vap_manifest_test.json \
    --compare \
    --output outputs/comparison_results.json
```

### 6.2 Output format

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

### 6.3 Can fix: trainer.py skip stage khi epochs = 0

Hien tai `VAPTrainer.train_stage()` se crash khi `epochs: 0` vi `OneCycleLR`
yeu cau `total_steps > 0`. Can them guard vao `train_stage()`:

```python
def train_stage(self, stage: int, start_epoch: int = 1):
    stage_cfg = self.config["training"][f"stage{stage}"]
    num_epochs = stage_cfg["epochs"]
    if num_epochs == 0:
        print(f"\n  Stage {stage}: skipped (epochs=0)")
        return
    # ... rest of method
```

Chua fix -> **khong the chay 1-stage hoac 2-stage configs**.

---

## 7. Paper Table Templates

### Table 1: Main Results

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
|  - HuTu          | 73.5     |  0.65    | 66.0  |
|  - PhoBERT       | 71.2     |  0.63    | 61.0  |
|  - Cross-Attn    | 74.1     |  0.66    | 68.0  |
|   (use GMU)      |          |          |       |
|  - Wav2Vec2-Vi   | 68.3     |  0.59    | 55.0  |
|   (use WavLM-EN) |          |          |       |
+------------------+----------+----------+-------+
```

### Table 3: HuTu Marker Impact

```
+------------------+--------+---------+-----------+-------+
| Marker Category  | Count  | Acc     | Acc       | Delta |
|                  |        | w/ HuTu | w/o HuTu  |       |
+------------------+--------+---------+-----------+-------+
| Yield (SFP)      | 1,234  | 82.1%   | 74.3%     | +7.8  |
| Hold (conj.)     | 2,456  | 78.5%   | 74.1%     | +4.4  |
| Backchannel      |   567  | 71.2%   | 59.8%     | +11.4 |
| Turn request     |   234  | 69.3%   | 63.2%     | +6.1  |
| No marker        | 15,678 | 74.8%   | 73.9%     | +0.9  |
+------------------+--------+---------+-----------+-------+
```

### Figure Ideas

1. **Training curves**: Loss/Acc across 3 stages (da co trong training_history.json)
2. **Latency CDF**: Cumulative distribution of EoT latency
3. **FPR-Latency tradeoff**: FPR at different latency thresholds
4. **Confusion matrix**: 3-class (shift/hold/BC) confusion matrix
5. **Qualitative examples**: 2-3 conversation excerpts showing predictions
6. **HuTu bar chart**: Per-marker accuracy improvement

---

## 8. Data Requirements

### 8.1 Hien tai

| Metric | Hien tai | Can cho paper |
|--------|---------|--------------|
| Total audio | ~5.5h (10 videos) | 20-50h (40-100 videos) |
| Train files | 33 segments | 150+ segments |
| Val files | 4 segments | 20+ segments |
| Test files | 5 segments | 30+ segments |
| Total frames | 1M | 4-10M |

### 8.2 Cach tang data

1. Tim them YouTube videos (podcast/phong van 2 nguoi tieng Viet):

```bash
# Them URLs vao scripts/urls.txt
# Chay lai pipeline
python scripts/00_download_audio.py --output data/audio
python scripts/00b_split_audio.py --input data/audio --output data/audio_split
# ... (pipeline tu dong)
```

2. Can dam bao **da dang**:
   - Bac/Trung/Nam dialect
   - Nam/nu speakers
   - Formal/informal conversations
   - Different topics

3. Annotate dialect info cho moi conversation (de dung `analyze_per_dialect`).

### 8.3 Data Split Strategy

Khi tang data, can chu y:
- Split theo **conversation** (khong phai theo window) de tranh data leakage
- Giu ty le 80/10/10 (train/val/test)
- Dam bao moi dialect co mat trong ca 3 splits

---

## 9. Checklist truoc khi nop paper

### Data
- [ ] 30+ gio audio (40+ videos)
- [ ] 3 dialects (Bac/Trung/Nam) co mat
- [ ] Test set >= 30 conversations
- [ ] Dialect annotation cho moi file

### Experiments
- [ ] Full model trained (3 stages)
- [ ] Random baseline
- [ ] Majority baseline
- [ ] Audio-only baseline (Stage 1)
- [ ] Text-only baseline
- [ ] Original VAP baseline (optional nhung tot)
- [ ] No-HuTu ablation
- [ ] Fusion type ablation (GMU, Bottleneck)
- [ ] Encoder ablation (WavLM vs Wav2Vec2-Vi)
- [ ] **Training strategy ablation (1-stage vs 2-stage vs 3-stage)**

### Metrics
- [ ] Tier 1: Frame Acc, F1, Perplexity, ECE
- [ ] Tier 2: Shift/Hold BA, BC F1, Shift AUC
- [ ] Tier 3: EoT Latency, FPR curves, VAQI
- [ ] Vietnamese analysis: Per-marker accuracy
- [ ] Per-dialect breakdown

### Statistics
- [ ] Bootstrap 95% CI cho moi metric
- [ ] Permutation test (p < 0.05) cho moi comparison
- [ ] Cohen's d effect sizes
- [ ] >= 30 test conversations cho power

### Figures
- [ ] Training curves (3 stages)
- [ ] Latency CDF
- [ ] FPR-Latency tradeoff
- [ ] Confusion matrix
- [ ] HuTu marker impact bar chart
- [ ] Qualitative examples

### Writing
- [ ] Abstract
- [ ] Introduction + motivation (Vietnamese turn-taking gap)
- [ ] Related work (VAP, Vietnamese NLP, multimodal)
- [ ] Method (architecture, HuTu, 3-stage training)
- [ ] Experiments (data, baselines, ablations)
- [ ] Results + Analysis
- [ ] Conclusion

---

## 10. Timeline de xuat

| Tuan | Task |
|------|------|
| 1-2 | Thu thap them data (30+ videos), chay pipeline |
| 3 | Train full model + baselines + ablations |
| 4 | Chay evaluation, statistical tests |
| 5-6 | Viet paper |
| 7 | Review noi bo, chinh sua |
| 8 | Nop paper |
