# Evaluation Metrics for MM-VAP-VI

## Hệ thống đánh giá toàn diện cho Multimodal Voice Activity Projection (Vietnamese)

---

## 1. Tổng quan Framework Đánh giá

### 1.1 Triết lý đánh giá

MM-VAP-VI cần được đánh giá ở **4 tầng** khác nhau, từ low-level đến application-level:

```
┌─────────────────────────────────────────────────────────────────┐
│  TẦNG 4: Application-Level (VAQI Score)                         │
│  "Model hoạt động tốt trong real-world voice agent không?"      │
│  → Composite score: interruptions + missed turns + latency      │
├─────────────────────────────────────────────────────────────────┤
│  TẦNG 3: Latency & Efficiency (Streaming Performance)           │
│  "Model phản ứng đủ nhanh cho real-time không?"                │
│  → MST, FPR, Latency curves, FLOPs, inference time             │
├─────────────────────────────────────────────────────────────────┤
│  TẦNG 2: Event-Based (Zero-shot Turn-Taking Events)             │
│  "Model dự đoán đúng các sự kiện turn-taking không?"           │
│  → Shift BA, Hold BA, BC F1, Short/Long BA                     │
│  → PRIMARY METRICS CHO PAPER                                   │
├─────────────────────────────────────────────────────────────────┤
│  TẦNG 1: Frame-Level (VAP Prediction Quality)                   │
│  "Model dự đoán đúng phân bố 256-class tại mỗi frame không?"  │
│  → Cross-entropy, Perplexity, Weighted F1, Calibration          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Metric cho từng mục đích

| Mục đích | Primary Metrics | Secondary Metrics |
|----------|----------------|-------------------|
| **Paper submission** | Shift/Hold BA, BC F1, Predict-Shift AUC | Perplexity, Short/Long BA |
| **So sánh với VAP gốc** | Shift/Hold BA (cùng evaluation protocol) | Cross-entropy loss |
| **So sánh với industry** | MST@FPR=5%, Latency-FPR AUC | VAQI, Interruption Rate |
| **Ablation study** | Shift BA (đại diện nhất) | BC F1 (sensitive to modality) |
| **Deployment readiness** | FPR < 5%, Latency P95 < 500ms | Inference FLOPs, Memory |

---

## 2. Tầng 1: Frame-Level Metrics

### 2.1 Cross-Entropy Loss

**Mục đích**: Đo chất lượng dự đoán phân bố xác suất trên 256 classes tại mỗi frame.

```
L_CE = -(1/N) × Σᵢ log(p(yᵢ | xᵢ))

Trong đó:
  N     = tổng số valid frames (không tính padding)
  yᵢ    = ground-truth class index ∈ [0, 255]
  p(yᵢ) = predicted probability cho class yᵢ
```

**Cách tính**:

```python
import torch.nn.functional as F

def compute_frame_ce(logits, labels, mask):
    """
    Args:
        logits: (B, T, 256) — raw model output
        labels: (B, T) — ground-truth class indices
        mask:   (B, T) — True for valid frames
    Returns:
        ce_loss: scalar
    """
    logits_flat = logits[mask]      # (N_valid, 256)
    labels_flat = labels[mask]      # (N_valid,)
    return F.cross_entropy(logits_flat, labels_flat)
```

**Kỳ vọng giá trị**:
- Random baseline: `log(256) ≈ 5.55`
- Majority-class baseline: phụ thuộc vào class distribution, ước tính ~3.5-4.0
- Good model: < 2.5
- Strong model: < 2.0

**Lưu ý**: CE loss bị ảnh hưởng bởi label smoothing khi training. Report CE trên val/test set KHÔNG dùng label smoothing để so sánh công bằng.

---

### 2.2 Perplexity

**Mục đích**: Biến đổi CE loss sang thang dễ diễn giải hơn.

```
PPL = exp(L_CE)

Diễn giải:
  PPL = 256 → random (model không biết gì)
  PPL = 1   → perfect (model chắc chắn 100%)
  PPL = 10  → model "phân vân" giữa ~10 classes trung bình
```

**Cách tính**:

```python
import math

def compute_perplexity(ce_loss):
    return math.exp(ce_loss)
```

**Ưu điểm so với CE loss**:
- Trực giác hơn: PPL = 15 nghĩa là model trung bình hesitate giữa 15 classes
- Dễ so sánh cross-model vì thang tuyệt đối

**Kỳ vọng**:
- Random: 256.0
- Majority-class: ~30-50
- Good model: ~8-12
- Strong model: ~5-8

---

### 2.3 Top-1 & Top-5 Accuracy

**Mục đích**: Phần trăm frames dự đoán đúng class chính xác.

```
Top-1 Acc = (1/N) × Σᵢ 𝟙[argmax(p(xᵢ)) = yᵢ]
Top-5 Acc = (1/N) × Σᵢ 𝟙[yᵢ ∈ top5(p(xᵢ))]
```

**Cách tính**:

```python
def compute_topk_accuracy(logits, labels, mask, k=1):
    """
    Args:
        logits: (B, T, 256)
        labels: (B, T)
        mask:   (B, T)
        k:      top-k
    Returns:
        accuracy: float
    """
    logits_flat = logits[mask]     # (N, 256)
    labels_flat = labels[mask]     # (N,)

    _, topk_preds = logits_flat.topk(k, dim=-1)  # (N, k)
    correct = topk_preds.eq(labels_flat.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()
```

**Kỳ vọng**:
- Top-1: ~40-55% (thấp vì 256 classes, long-tail distribution)
- Top-5: ~65-80%

**Lưu ý**: Top-1 accuracy bị bias bởi frequent classes (hold, silence). Cần Weighted F1 để đánh giá cân bằng hơn.

---

### 2.4 Weighted F1

**Mục đích**: F1-score trung bình có trọng số theo frequency, tránh bias từ majority class.

```
Weighted F1 = Σ_c (n_c / N) × F1_c

Trong đó:
  c   = class index [0-255]
  n_c = số frames thuộc class c
  F1_c = 2 × Precision_c × Recall_c / (Precision_c + Recall_c)
```

**Cách tính**:

```python
from sklearn.metrics import f1_score

def compute_weighted_f1(logits, labels, mask):
    preds = logits[mask].argmax(dim=-1).cpu().numpy()
    truth = labels[mask].cpu().numpy()
    return f1_score(truth, preds, average='weighted', zero_division=0)
```

**Kỳ vọng**: 0.35-0.55

**Report thêm**: Macro F1 (trung bình không trọng số) để xem model có bỏ qua rare classes không. Nếu Macro F1 << Weighted F1 → model chỉ tốt ở frequent classes.

---

### 2.5 Calibration Metrics (Novel — chưa có paper VAP nào report)

**Mục đích**: Đo mức độ tin cậy của xác suất model output. Quan trọng cho decision-making trong streaming (threshold-based).

#### 2.5.1 Expected Calibration Error (ECE)

```
ECE = Σ_b (|B_b| / N) × |acc(B_b) - conf(B_b)|

Trong đó:
  B_b      = tập frames có confidence rơi vào bin b
  acc(B_b) = tỷ lệ dự đoán đúng trong bin b
  conf(B_b)= confidence trung bình trong bin b
  N bins = 15 (recommended)
```

Đo cho **aggregated event probabilities**:

```python
import numpy as np

def compute_ece(probs, labels, n_bins=15):
    """
    Compute ECE for binary event (e.g., shift vs non-shift).

    Args:
        probs:  (N,) — predicted P(shift) ∈ [0, 1]
        labels: (N,) — binary ground-truth (1=shift, 0=non-shift)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i+1])
        if mask.sum() == 0:
            continue

        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)

    return ece
```

**Kỳ vọng**: ECE < 0.05 là well-calibrated, < 0.10 là acceptable.

**Tại sao quan trọng cho MM-VAP-VI**: Streaming inference dùng threshold (P(shift) > θ → respond). Nếu model poorly calibrated, θ không có ý nghĩa nhất quán → behavior không ổn định.

#### 2.5.2 Brier Score

```
BS = (1/N) × Σᵢ (pᵢ - oᵢ)²

Trong đó:
  pᵢ = predicted probability cho event
  oᵢ = actual outcome (0 hoặc 1)
```

**Decomposition**:

```
BS = Reliability - Resolution + Uncertainty
     (calibration)  (sharpness)  (inherent noise)
```

- **Reliability↓**: model confidence khớp actual frequency
- **Resolution↑**: model phân biệt tốt giữa events
- **Uncertainty**: constant, phụ thuộc data

#### 2.5.3 Reliability Diagram

Visualization quan trọng nhất cho calibration:

```
1.0 ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
    │              ·   /│
    │           ·    /  │
    │        ·     /    │  ── Perfect calibration (diagonal)
    │     ·      /      │  ·· Actual model
    │   ·      /        │
    │ ·      /          │
0.0 ├──────────────────┤
   0.0    Confidence   1.0
```

Report cho: P(shift), P(hold), P(backchannel)

---

## 3. Tầng 2: Event-Based Metrics (Primary cho Paper)

### 3.1 Event Extraction Protocol (theo VAP gốc)

**Quan trọng**: Tất cả events được extract từ ground-truth voice activity, không phải từ model predictions. Model chỉ được đánh giá tại các event locations.

```
┌─────────────────────────────────────────────────────────────────┐
│  EVENT EXTRACTION (từ voice activity matrix, ground-truth)       │
│                                                                  │
│  Input: va_matrix (2, T), frame_rate = 50                        │
│                                                                  │
│  Điều kiện chung:                                                │
│  - min_silence = 0.25s (12 frames) — silence tối thiểu giữa     │
│    2 utterances để tính là event                                 │
│  - min_utterance = 1.0s (50 frames) — utterance tối thiểu       │
│    trước/sau silence để tính event (loại bỏ noise/short speech)  │
│                                                                  │
│  ═══════════════════════════════════════════════════             │
│  SHIFT events:                                                   │
│    SP_a nói ≥ 1s → silence ≥ 0.25s → SP_b nói ≥ 1s             │
│    (SP_a ≠ SP_b → speaker changed)                              │
│                                                                  │
│    Evaluation point: 0.05s (2-3 frames) sau silence onset        │
│    → Đọc p_now tại điểm này                                     │
│    → Nếu p_now favor SP_b → correct shift prediction             │
│                                                                  │
│  ═══════════════════════════════════════════════════             │
│  HOLD events:                                                    │
│    SP_a nói ≥ 1s → silence ≥ 0.25s → SP_a nói lại ≥ 1s         │
│    (Cùng speaker → within-turn pause)                            │
│                                                                  │
│    Evaluation point: 0.05s sau silence onset                     │
│    → Đọc p_now tại điểm này                                     │
│    → Nếu p_now favor SP_a → correct hold prediction              │
│                                                                  │
│  ═══════════════════════════════════════════════════             │
│  BACKCHANNEL events:                                             │
│    SP_a đang nói → SP_b nói ngắn (< 1s)                         │
│    → SP_a vẫn giữ lượt (active sau BC)                          │
│                                                                  │
│    BC conditions (VAP standard):                                 │
│      - SP_b duration < 1.0s                                      │
│      - SP_b silence trước BC ≥ 1.0s                              │
│      - SP_b silence sau BC ≥ 2.0s                                │
│    Positive: 1s trước BC onset                                   │
│    Negative: random non-BC points                                │
│                                                                  │
│  ═══════════════════════════════════════════════════             │
│  SHORT/LONG events:                                              │
│    Tại onset của listener activity:                              │
│      - SHORT: response < 1s (backchannel)                        │
│      - LONG: response ≥ 1s (actual turn)                         │
│    → Model cần phân biệt ngay từ onset                           │
│                                                                  │
│  ═══════════════════════════════════════════════════             │
│  OVERLAP events (optional):                                      │
│    Cả SP_a và SP_b active cùng lúc > 200ms                      │
│    Types: cooperative (BC), competitive (interruption)            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 p_now Computation

**p_now** là metric trung tâm của VAP — xác suất speaker tiếp theo tại mỗi frame.

```python
import torch
import numpy as np

def compute_p_now(probs, num_bins=4):
    """
    Compute next-speaker probability from VAP 256-class distribution.

    Args:
        probs: (T, 256) — softmax probabilities per frame

    Returns:
        p_now: (T,) — P(next_speaker = SP1)
                       (P(SP2) = 1 - p_now)

    Method:
        Sum probabilities of all classes where SP1 is more active
        than SP2 in the near-future bins (bins 0-1).
    """
    T = probs.shape[0]
    p_sp1 = torch.zeros(T)

    for class_idx in range(256):
        bits = []
        val = class_idx
        for _ in range(num_bins * 2):
            bits.append(val % 2)
            val //= 2

        sp1_bins = bits[:num_bins]
        sp2_bins = bits[num_bins:]

        # Near-future activity (bins 0-1: 0-600ms)
        sp1_near = sum(sp1_bins[:2])
        sp2_near = sum(sp2_bins[:2])

        if sp1_near > sp2_near:
            p_sp1 += probs[:, class_idx]
        elif sp1_near == sp2_near:
            p_sp1 += 0.5 * probs[:, class_idx]

    return p_sp1  # (T,)
```

### 3.3 Balanced Accuracy (Shift/Hold)

**Metric chính cho paper**, so sánh trực tiếp với VAP literature.

```
Balanced Accuracy = (Recall_shift + Recall_hold) / 2
                  = (TP_shift/(TP_shift+FN_shift) + TP_hold/(TP_hold+FN_hold)) / 2
```

**Tại sao Balanced Accuracy thay vì Accuracy?**
- Shift events thường ít hơn hold events (ratio ~1:2 đến 1:3)
- Regular accuracy sẽ bias toward hold predictions
- Balanced Accuracy = trung bình recall 2 classes → công bằng

```python
from sklearn.metrics import balanced_accuracy_score

def evaluate_shift_hold(events, p_now_at_events):
    """
    Args:
        events: List[Dict] with keys:
            - type: "shift" or "hold"
            - frame: evaluation frame index
            - prev_speaker: 0 or 1
            - next_speaker: 0 or 1 (=prev_speaker for hold)

        p_now_at_events: List[float] — p_now value at each event

    Returns:
        balanced_accuracy: float
        per_class_recall: Dict
    """
    y_true = []  # 1 = shift, 0 = hold
    y_pred = []

    for event, p_now in zip(events, p_now_at_events):
        if event["type"] == "shift":
            y_true.append(1)
            # Correct if p_now favors next_speaker (different from prev)
            if event["prev_speaker"] == 0:
                # prev=SP1, next=SP2 → p_now should be LOW (favor SP2)
                y_pred.append(1 if p_now < 0.5 else 0)
            else:
                y_pred.append(1 if p_now > 0.5 else 0)
        else:  # hold
            y_true.append(0)
            if event["prev_speaker"] == 0:
                # prev=SP1, next=SP1 → p_now should be HIGH (favor SP1)
                y_pred.append(0 if p_now > 0.5 else 1)
            else:
                y_pred.append(0 if p_now < 0.5 else 1)

    ba = balanced_accuracy_score(y_true, y_pred)

    # Per-class recall
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    shift_recall = y_pred[y_true == 1].mean() if (y_true == 1).sum() > 0 else 0
    hold_recall = (1 - y_pred[y_true == 0]).mean() if (y_true == 0).sum() > 0 else 0

    return {
        "balanced_accuracy": round(ba, 4),
        "shift_recall": round(shift_recall, 4),
        "hold_recall": round(hold_recall, 4),
        "num_shifts": int((y_true == 1).sum()),
        "num_holds": int((y_true == 0).sum()),
    }
```

**Reference values** (từ literature):

| Model | Dataset | Shift/Hold BA |
|-------|---------|---------------|
| VAP (CPC, EN) | Switchboard | ~78% |
| Multilingual VAP (MMS, ZH) | CallHome Mandarin | ~84% |
| Multilingual VAP (MMS, JA) | CallHome Japanese | ~76% |
| Lla-VAP (Llama + VAP) | Switchboard | ~80% |
| **MM-VAP-VI (target)** | Vietnamese Podcast | **~78-85%** |

---

### 3.4 Predict-Shift (Anticipatory Prediction)

**Mục đích**: Model có thể dự đoán shift TRƯỚC KHI silence xảy ra không? (Giống như humans anticipate turn-ends ~340ms trước.)

```
┌─────────────────────────────────────────────────────────────────┐
│  PREDICT-SHIFT EVALUATION                                        │
│                                                                  │
│  Positive examples (predict_shift_pos):                          │
│    SP_a đang nói → sẽ có shift                                  │
│    Evaluation region: 3s trước silence onset                     │
│    → Model should have INCREASING p(shift)                       │
│                                                                  │
│  Negative examples (predict_shift_neg):                          │
│    SP_a đang nói → sẽ có hold (continue)                        │
│    Evaluation region: 3s trước pause onset                       │
│    → Model should have LOW p(shift)                              │
│                                                                  │
│  Timeline example:                                               │
│                                                                  │
│  Positive:                                                       │
│    SP_a: ═══════════════════╗                                    │
│                   ↑         ↑                                    │
│                eval_start  silence → SP_b starts                 │
│                 (-3s)      (0s)                                  │
│    p(shift):  0.2 → 0.4 → 0.6 → 0.8  (increasing = good)      │
│                                                                  │
│  Negative:                                                       │
│    SP_a: ═══════════════════╗    ╔═══════                        │
│                   ↑         ↑    ↑                               │
│                eval_start  pause  SP_a resumes                   │
│    p(shift):  0.2 → 0.2 → 0.3 → 0.2  (stable low = good)      │
│                                                                  │
│  Metric: AUC-ROC hoặc AUC-PR trên tập {pos, neg}               │
│  (mỗi frame trong evaluation region là 1 sample)                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_predict_shift(pos_probs, neg_probs):
    """
    Args:
        pos_probs: List[np.array] — p(shift) values for each positive region
        neg_probs: List[np.array] — p(shift) values for each negative region
    """
    all_probs = np.concatenate(
        [np.concatenate(pos_probs), np.concatenate(neg_probs)]
    )
    all_labels = np.concatenate([
        np.ones(sum(len(p) for p in pos_probs)),
        np.zeros(sum(len(p) for p in neg_probs))
    ])

    return {
        "auc_roc": round(roc_auc_score(all_labels, all_probs), 4),
        "auc_pr": round(average_precision_score(all_labels, all_probs), 4),
    }
```

---

### 3.5 Backchannel Prediction F1

**Mục đích**: Model dự đoán đúng khi listener sắp nói backchannel.

```python
from sklearn.metrics import precision_recall_fscore_support

def evaluate_backchannel(pos_probs, neg_probs, threshold=0.3):
    """
    Evaluate backchannel prediction.

    Args:
        pos_probs: List[float] — p(bc) at positive BC prediction points
        neg_probs: List[float] — p(bc) at negative (random non-BC) points
        threshold: decision threshold

    Note: threshold mặc định 0.3 (thấp hơn 0.5) vì:
      - BC events hiếm (~3-8% frames)
      - Recall quan trọng hơn precision cho BC
      - Trong ứng dụng, false positive BC (nói "ừ" sai lúc) ít hại hơn
        false negative (bỏ qua cơ hội tương tác)
    """
    all_probs = np.concatenate([pos_probs, neg_probs])
    all_labels = np.concatenate([
        np.ones(len(pos_probs)),
        np.zeros(len(neg_probs))
    ])

    preds = (all_probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average='binary'
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "threshold": threshold,
        "num_positive": len(pos_probs),
        "num_negative": len(neg_probs),
    }
```

**Report tại nhiều thresholds**: θ ∈ {0.2, 0.3, 0.4, 0.5} + AUC-PR curve.

---

### 3.6 Short vs. Long Balanced Accuracy

**Mục đích**: Tại onset của listener response, model có phân biệt được backchannel (short) vs actual turn-take (long) ngay lập tức không?

```
Scenario: Listener bắt đầu nói tại frame t.
  → SHORT: listener nói < 1s rồi dừng (backchannel)
  → LONG:  listener nói ≥ 1s (taking the turn)

Model cần quyết định TẠI FRAME t (không có future info).
Metric: Balanced Accuracy(short, long)
```

**Quan trọng cho Vietnamese**: Tiếng Việt có nhiều backchannels ngắn ("ừ", "dạ", "vâng") → model cần phân biệt sớm.

---

### 3.7 Tổng hợp Event Metrics

```
┌──────────────────────────────────────────────────────────────────┐
│  EVENT METRICS SUMMARY TABLE (cho paper)                          │
│                                                                   │
│  ┌────────────────┬──────────┬───────┬───────┬──────────────┐    │
│  │ Model          │ S/H BA ↑ │ BC F1↑│ S/L BA│ PredShift AUC│    │
│  ├────────────────┼──────────┼───────┼───────┼──────────────┤    │
│  │ Random         │  50.0%   │  0.0% │ 50.0% │    0.500     │    │
│  │ VAD-only       │ ~65.0%   │  N/A  │  N/A  │     N/A      │    │
│  │ VAP-Wav2Vec2   │  ~78%    │ ~45%  │ ~68%  │   ~0.72      │    │
│  │  (audio-only)  │          │       │       │              │    │
│  │ Text-only      │  ~72%    │ ~35%  │ ~65%  │   ~0.68      │    │
│  │ MM-VAP-VI      │  ~82%    │ ~52%  │ ~73%  │   ~0.78      │    │
│  │  (full model)  │          │       │       │              │    │
│  └────────────────┴──────────┴───────┴───────┴──────────────┘    │
│                                                                   │
│  Statistical significance: Bootstrap 95% CI hoặc paired t-test   │
│  trên per-conversation scores.                                    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Tầng 3: Latency & Efficiency Metrics

### 4.1 End-of-Turn Detection Latency

**Mục đích**: Đo thời gian từ khi speaker thực sự hết lượt đến khi model detect shift.

```
┌─────────────────────────────────────────────────────────────────┐
│  EOT LATENCY MEASUREMENT                                        │
│                                                                  │
│  Speaker:  ═══════════════════╗                                  │
│                               ↑                                  │
│                          actual EOT (t*)                         │
│                                                                  │
│  Model p(shift):                                                 │
│            ....0.2...0.3...0.5...0.7...0.8....                   │
│                               ↑          ↑                       │
│                             t*          t (first frame > θ)      │
│                                                                  │
│  Latency = (t - t*) × 20ms                                      │
│                                                                  │
│  Negative latency = model predicted BEFORE EOT (anticipatory)    │
│  Positive latency = model predicted AFTER EOT (reactive)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
def compute_eot_latency(shift_events, p_shift_series, threshold=0.5,
                         frame_rate=50):
    """
    Args:
        shift_events: List[Dict] with 'eot_frame' — actual end-of-turn frame
        p_shift_series: (T,) — continuous p(shift) time series
        threshold: detection threshold
    Returns:
        latencies_ms: List[float]
        stats: Dict with median, mean, P90, P95, early_rate
    """
    latencies_ms = []

    for event in shift_events:
        eot_frame = event["eot_frame"]

        # Search forward from 1s before EOT to 2s after
        search_start = max(0, eot_frame - frame_rate)  # -1s
        search_end = min(len(p_shift_series), eot_frame + 2 * frame_rate)  # +2s

        detected_frame = None
        for t in range(search_start, search_end):
            if p_shift_series[t] >= threshold:
                detected_frame = t
                break

        if detected_frame is not None:
            latency_frames = detected_frame - eot_frame
            latencies_ms.append(latency_frames * (1000 / frame_rate))

    latencies = np.array(latencies_ms)

    return {
        "median_ms": round(np.median(latencies), 1),
        "mean_ms": round(np.mean(latencies), 1),
        "std_ms": round(np.std(latencies), 1),
        "p90_ms": round(np.percentile(latencies, 90), 1),
        "p95_ms": round(np.percentile(latencies, 95), 1),
        "early_detection_rate": round((latencies < 0).mean(), 4),
        "detection_rate": round(len(latencies) / len(shift_events), 4),
        "threshold": threshold,
        "num_events": len(shift_events),
        "num_detected": len(latencies),
    }
```

**Target values**:

| Metric | Human-like | Acceptable | Poor |
|--------|-----------|------------|------|
| Median latency | < 200ms | 200-500ms | > 500ms |
| P95 latency | < 500ms | 500-1000ms | > 1000ms |
| Early detection rate | > 30% | 10-30% | < 10% |
| Detection rate | > 95% | 90-95% | < 90% |

---

### 4.2 False Endpoint Rate (FPR)

**Mục đích**: Bao nhiêu phần trăm hold pauses bị nhầm là turn-ends?

```
FPR = |hold events where p(shift) > θ| / |total hold events|
```

```python
def compute_fpr(hold_events, p_shift_at_holds, threshold=0.5):
    """
    Args:
        hold_events: List[Dict] — hold events from extraction
        p_shift_at_holds: List[float] — p(shift) at each hold event
    """
    false_positives = sum(1 for p in p_shift_at_holds if p >= threshold)
    fpr = false_positives / len(hold_events) if hold_events else 0

    return {
        "fpr": round(fpr, 4),
        "false_positives": false_positives,
        "total_holds": len(hold_events),
        "threshold": threshold,
    }
```

**Target**: FPR < 5% tại θ tối ưu. Report tại θ ∈ {0.3, 0.4, 0.5, 0.6, 0.7}.

---

### 4.3 Mean Shift Time (MST) vs FPR Curve

**Metric chính từ Krisp** — cho phép so sánh latency-accuracy tradeoff.

```
┌─────────────────────────────────────────────────────────────────┐
│  MST vs FPR CURVE                                                │
│                                                                  │
│  Sweep θ from 0.1 to 0.9 (step 0.05)                            │
│  At each θ:                                                      │
│    - MST(θ) = mean latency for correctly detected shifts         │
│    - FPR(θ) = false positive rate on holds                       │
│                                                                  │
│  Plot:                                                           │
│    MST (ms)                                                      │
│    800 ─┐                                                        │
│         │  ╲                                                     │
│    600 ─┤   ╲  (high threshold = slow but safe)                  │
│         │    ╲                                                   │
│    400 ─┤     ╲                                                  │
│         │      ╲─── Model A (better)                             │
│    200 ─┤       ╲                                                │
│         │        ╲── Model B (worse)                             │
│     0 ──┼──┬──┬──┬──┬──┬──→ FPR                                 │
│         0  5  10 15 20 25 (%)                                    │
│                                                                  │
│  Single-number metric:                                           │
│    Area Under MST-FPR Curve (AU-MFC)                             │
│    Lower = better (fast response with low false positives)       │
│                                                                  │
│  MST @ FPR=5%:                                                   │
│    Industry standard: report MST at fixed FPR for fair compare   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
def compute_mst_fpr_curve(shift_events, hold_events, p_shift_series,
                           frame_rate=50, thresholds=None):
    """
    Compute MST vs FPR curve over multiple thresholds.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    curve_points = []

    for theta in thresholds:
        # MST: mean latency for detected shifts
        latencies = []
        for event in shift_events:
            eot = event["eot_frame"]
            search_end = min(len(p_shift_series), eot + 5 * frame_rate)
            for t in range(eot, search_end):
                if p_shift_series[t] >= theta:
                    latencies.append((t - eot) * (1000 / frame_rate))
                    break

        mst = np.mean(latencies) if latencies else float('inf')

        # FPR: false positive rate on holds
        fp = sum(1 for e in hold_events
                 if p_shift_series[e["eval_frame"]] >= theta)
        fpr = fp / len(hold_events) if hold_events else 0

        curve_points.append({
            "threshold": round(theta, 2),
            "mst_ms": round(mst, 1),
            "fpr": round(fpr, 4),
            "detection_rate": round(len(latencies) / len(shift_events), 4),
        })

    # AUC (trapezoidal integration)
    fprs = [p["fpr"] for p in curve_points]
    msts = [p["mst_ms"] for p in curve_points]
    au_mfc = np.trapz(msts, fprs)

    # MST @ FPR=5%
    mst_at_5pct = None
    for p in curve_points:
        if p["fpr"] <= 0.05:
            mst_at_5pct = p["mst_ms"]
            break

    return {
        "curve": curve_points,
        "au_mfc": round(au_mfc, 2),
        "mst_at_fpr_5pct": mst_at_5pct,
    }
```

---

### 4.4 Efficiency Metrics (Model Size & Speed)

```
┌─────────────────────────────────────────────────────────────────┐
│  EFFICIENCY METRICS                                              │
│                                                                  │
│  1. Model Parameters                                             │
│     - Total params (all)                                         │
│     - Trainable params                                           │
│     - Frozen params                                              │
│     Report: "237M total, 6.9M trainable"                         │
│                                                                  │
│  2. Inference FLOPs (per frame)                                  │
│     - Acoustic encoder:    ~X MFLOPs                             │
│     - Linguistic encoder:  ~Y MFLOPs (amortized per 25 frames)   │
│     - Fusion + Transformer: ~Z MFLOPs                            │
│     - Total per frame:     ~(X + Y/25 + Z) MFLOPs               │
│                                                                  │
│  3. Inference Latency (per frame, batch=1)                       │
│     - GPU (RTX 3090): target < 10ms/frame                        │
│     - CPU (modern x86): target < 50ms/frame                      │
│     Measure: mean ± std over 1000 frames after warmup             │
│                                                                  │
│  4. Real-Time Factor (RTF)                                       │
│     RTF = processing_time / audio_duration                       │
│     RTF < 1.0 → real-time capable                                │
│     Target: RTF < 0.5 on GPU, RTF < 1.0 on CPU                  │
│                                                                  │
│  5. Memory                                                       │
│     - Peak GPU memory (batch=1, streaming): target < 2GB         │
│     - KV cache size for 10s context                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
import time
import torch

def benchmark_inference(model, sample_audio, num_frames=1000,
                         warmup=100, device="cuda"):
    """
    Benchmark single-frame inference latency.
    """
    model.eval()
    model.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model.process_frame(sample_audio.to(device))

    # Benchmark
    torch.cuda.synchronize() if device == "cuda" else None
    latencies = []

    for _ in range(num_frames):
        start = time.perf_counter()
        with torch.no_grad():
            model.process_frame(sample_audio.to(device))
        torch.cuda.synchronize() if device == "cuda" else None
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)

    return {
        "mean_ms": round(latencies.mean(), 2),
        "std_ms": round(latencies.std(), 2),
        "p50_ms": round(np.percentile(latencies, 50), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
        "p99_ms": round(np.percentile(latencies, 99), 2),
        "rtf": round(latencies.mean() / 20.0, 4),  # 20ms per frame
        "device": device,
    }
```

---

## 5. Tầng 4: Application-Level Metrics

### 5.1 VAQI — Voice Agent Quality Index (Deepgram-inspired)

**Mục đích**: Một con số duy nhất đánh giá tổng thể chất lượng turn-taking trong bối cảnh voice agent.

```
VAQI = 100 × (1 - [0.4 × I + 0.4 × M + 0.2 × L])

Trong đó:
  I = Interruption Score ∈ [0, 1]
      = (số lần model "cắt lời" user) / (tổng turns)
      Cắt lời = model predict shift khi user chưa xong

  M = Missed Response Score ∈ [0, 1]
      = (số lần model không respond khi user xong) / (tổng shifts)
      Missed = model predict hold khi user đã yield

  L = Latency Score ∈ [0, 1]
      = log(1 + median_latency_ms) / log(1 + max_acceptable_latency)
      max_acceptable = 2000ms

VAQI ∈ [0, 100]:
  90-100: Excellent (near-human turn-taking)
  70-89:  Good (minor issues)
  50-69:  Fair (noticeable interruptions/delays)
  <50:    Poor (frequent issues)
```

```python
import math

def compute_vaqi(shift_events, hold_events, p_shift_series,
                  threshold=0.5, frame_rate=50, max_latency=2000):
    """
    Compute Voice Agent Quality Index.

    Adapts Deepgram's VAQI for offline evaluation on test set.
    """
    # I: Interruption rate (hold events classified as shift)
    interruptions = sum(
        1 for e in hold_events
        if p_shift_series[e["eval_frame"]] >= threshold
    )
    I = interruptions / len(hold_events) if hold_events else 0

    # M: Missed response rate (shift events not detected within 1s)
    missed = 0
    latencies = []
    for event in shift_events:
        eot = event["eot_frame"]
        detected = False
        for t in range(eot, min(len(p_shift_series), eot + frame_rate)):
            if p_shift_series[t] >= threshold:
                latencies.append((t - eot) * (1000 / frame_rate))
                detected = True
                break
        if not detected:
            missed += 1
    M = missed / len(shift_events) if shift_events else 0

    # L: Latency score
    median_latency = np.median(latencies) if latencies else max_latency
    L = math.log(1 + median_latency) / math.log(1 + max_latency)
    L = min(L, 1.0)

    vaqi = 100 * (1 - (0.4 * I + 0.4 * M + 0.2 * L))

    return {
        "vaqi": round(vaqi, 1),
        "interruption_rate": round(I, 4),
        "missed_response_rate": round(M, 4),
        "latency_score": round(L, 4),
        "median_latency_ms": round(median_latency, 1),
    }
```

---

### 5.2 Sequence-Based EoT Evaluation (Deepgram-style)

**Mục đích**: Đánh giá turn-taking trong context hoàn chỉnh conversation, không chỉ isolated events.

```
┌─────────────────────────────────────────────────────────────────┐
│  SEQUENCE-BASED EVALUATION                                       │
│                                                                  │
│  Ý tưởng: Chèn [EoT] token vào transcript tại mỗi predicted    │
│  và actual turn boundary, rồi dùng modified Levenshtein để      │
│  match.                                                          │
│                                                                  │
│  Ground truth: "Tôi muốn hỏi [EoT] Vâng được ạ [EoT]"         │
│  Prediction:   "Tôi muốn hỏi nhé [EoT] Vâng [EoT] được ạ"     │
│                                                                  │
│  Alignment:                                                      │
│    [EoT]₁ (GT) ↔ [EoT]₁ (Pred) → Match (latency = +1 word)    │
│    [EoT]₂ (GT) ↔ [EoT]₂ (Pred) → Early trigger (1 word early) │
│                                                                  │
│  Metrics:                                                        │
│    - EoT Precision: % predicted [EoT] that match actual          │
│    - EoT Recall: % actual [EoT] that were predicted              │
│    - EoT F1                                                      │
│    - Mean position error (in words or ms)                        │
│                                                                  │
│  Ưu điểm: Đánh giá trong context → realistic hơn isolated event │
│  Nhược: Cần ASR transcript → dependent on ASR quality            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Vietnamese-Specific Metrics

### 6.1 Discourse Marker Analysis

**Mục đích**: Đo mức độ model leverage discourse markers tiếng Việt.

```python
def analyze_marker_impact(events, p_shift_series, transcripts,
                           marker_sets):
    """
    So sánh model performance tại events có/không có discourse marker.

    Args:
        events: List[Dict] — shift/hold events
        transcripts: text tại mỗi event
        marker_sets: Dict of marker categories
    """
    with_marker = {"shift": [], "hold": []}
    without_marker = {"shift": [], "hold": []}

    for event in events:
        text = event.get("text", "").lower()
        has_marker = any(
            m in text.split()
            for markers in marker_sets.values()
            for m in markers
        )

        bucket = with_marker if has_marker else without_marker
        correct = event_is_correct(event, p_shift_series)
        bucket[event["type"]].append(correct)

    return {
        "with_marker": {
            "shift_accuracy": np.mean(with_marker["shift"]),
            "hold_accuracy": np.mean(with_marker["hold"]),
            "count": len(with_marker["shift"]) + len(with_marker["hold"]),
        },
        "without_marker": {
            "shift_accuracy": np.mean(without_marker["shift"]),
            "hold_accuracy": np.mean(without_marker["hold"]),
            "count": len(without_marker["shift"]) + len(without_marker["hold"]),
        },
        "marker_benefit": {
            "shift_delta": (np.mean(with_marker["shift"])
                          - np.mean(without_marker["shift"])),
            "hold_delta": (np.mean(with_marker["hold"])
                          - np.mean(without_marker["hold"])),
        }
    }
```

### 6.2 Per-Dialect Performance

```
┌─────────────────────────────────────────────────────────────────┐
│  DIALECT ANALYSIS                                                │
│                                                                  │
│  Nếu metadata có dialect tags (Bắc/Trung/Nam):                  │
│                                                                  │
│  ┌──────────┬──────────┬───────┬──────────┐                     │
│  │ Dialect  │ S/H BA ↑ │ BC F1↑│ MST (ms) │                     │
│  ├──────────┼──────────┼───────┼──────────┤                     │
│  │ Bắc      │          │       │          │                     │
│  │ Trung    │          │       │          │                     │
│  │ Nam      │          │       │          │                     │
│  │ Mixed    │          │       │          │                     │
│  └──────────┴──────────┴───────┴──────────┘                     │
│                                                                  │
│  Kỳ vọng: Nam có thể khó hơn (nhiều backchannel particles hơn,  │
│  intonation patterns khác biệt lớn với Bắc)                     │
│                                                                  │
│  Lưu ý: Chỉ có ý nghĩa nếu đủ data per dialect (>2h mỗi loại) │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Tonal F0 Confusion Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│  TONAL F0 ANALYSIS                                               │
│                                                                  │
│  Mục đích: Model có bị confuse giữa lexical tone và             │
│  intonational turn cues không?                                   │
│                                                                  │
│  Method:                                                         │
│    1. Extract F0 contour tại turn boundaries                     │
│    2. Phân loại theo tone cuối (sắc↑, huyền↓, nặng↓., hỏi↓↑...)│
│    3. So sánh model accuracy theo tone cuối                      │
│                                                                  │
│  Hypothesis:                                                     │
│    - Falling tones (huyền, nặng) → dễ predict shift              │
│      (trùng với falling intonation = universal turn cue)         │
│    - Rising tones (sắc, hỏi) → khó hơn                          │
│      (rising = typical hold/question cue trong các ngôn ngữ khác │
│       nhưng trong Vietnamese chỉ là lexical tone)                │
│                                                                  │
│  Output table:                                                   │
│  ┌──────────┬────────────┬──────────┬────────────────┐          │
│  │ Final    │ Shift Acc  │ Hold Acc │ Confusion Type │          │
│  │ Tone     │            │          │                │          │
│  ├──────────┼────────────┼──────────┼────────────────┤          │
│  │ Ngang    │            │          │                │          │
│  │ Huyền    │            │          │                │          │
│  │ Sắc      │            │          │                │          │
│  │ Hỏi      │            │          │                │          │
│  │ Ngã      │            │          │                │          │
│  │ Nặng     │            │          │                │          │
│  └──────────┴────────────┴──────────┴────────────────┘          │
│                                                                  │
│  Nếu acoustic model (SSL) disentangle tốt:                      │
│    → performance tương đối đều qua các tones                     │
│  Nếu model bị confuse:                                           │
│    → rising tones (sắc, hỏi) có shift accuracy thấp hơn          │
│                                                                  │
│  Reference: Mandarin (cũng tonal) đạt 84% BA trong              │
│  Multilingual VAP → evidence rằng SSL models CÓ THỂ disentangle │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Statistical Significance

### 7.1 Bootstrap Confidence Intervals

```python
def bootstrap_ci(metric_fn, data, n_bootstrap=1000, ci=0.95, seed=42):
    """
    Compute bootstrap confidence interval for any metric.

    Args:
        metric_fn: function that takes data and returns scalar metric
        data: input data (list of per-conversation results)
        n_bootstrap: number of bootstrap samples
        ci: confidence level
    """
    rng = np.random.RandomState(seed)
    bootstrap_values = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(data), size=len(data), replace=True)
        sample = [data[i] for i in idx]
        bootstrap_values.append(metric_fn(sample))

    lower = np.percentile(bootstrap_values, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_values, (1 + ci) / 2 * 100)

    return {
        "mean": round(np.mean(bootstrap_values), 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "ci_level": ci,
    }
```

### 7.2 Paired Permutation Test

```python
def permutation_test(scores_a, scores_b, n_permutations=10000, seed=42):
    """
    Per-conversation paired permutation test.

    H0: Model A and Model B have same performance.
    H1: They differ.

    Args:
        scores_a: per-conversation metric for Model A
        scores_b: per-conversation metric for Model B
    """
    rng = np.random.RandomState(seed)
    assert len(scores_a) == len(scores_b)

    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    diffs = np.array(scores_a) - np.array(scores_b)

    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = np.mean(diffs * signs)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = count / n_permutations

    return {
        "observed_diff": round(observed_diff, 4),
        "p_value": round(p_value, 4),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
    }
```

**Report format cho paper**:

```
Shift/Hold BA: 82.3% ± 1.2% (95% CI: [80.9%, 83.7%])
Improvement over audio-only: +4.1% (p < 0.01, paired permutation test)
```

---

## 8. Complete Evaluation Pipeline

### 8.1 Unified Evaluator Class

```python
class MMVAPEvaluator:
    """
    Complete evaluation pipeline for MM-VAP-VI.

    Usage:
        evaluator = MMVAPEvaluator(model, test_loader, device="cuda")
        results = evaluator.run_full_evaluation()
        evaluator.generate_report(results, output_dir="results/")
    """

    def __init__(self, model, test_loader, frame_rate=50, device="cuda"):
        self.model = model
        self.test_loader = test_loader
        self.frame_rate = frame_rate
        self.device = device

    def run_full_evaluation(self):
        """Run all evaluation tiers."""
        results = {}

        # Step 1: Compute predictions for entire test set
        all_logits, all_labels, all_masks = self._predict_all()
        all_probs = torch.softmax(all_logits, dim=-1)

        # Tier 1: Frame-level
        results["frame_level"] = {
            "ce_loss": compute_frame_ce(all_logits, all_labels, all_masks),
            "perplexity": compute_perplexity(ce_loss),
            "top1_accuracy": compute_topk_accuracy(all_logits, all_labels, all_masks, k=1),
            "top5_accuracy": compute_topk_accuracy(all_logits, all_labels, all_masks, k=5),
            "weighted_f1": compute_weighted_f1(all_logits, all_labels, all_masks),
            "macro_f1": compute_macro_f1(all_logits, all_labels, all_masks),
        }

        # Tier 2: Event-based
        events = self._extract_events()  # from ground-truth VA
        p_now = compute_p_now(all_probs)
        p_shift = compute_p_shift(all_probs)
        p_bc = compute_p_bc(all_probs)

        results["event_based"] = {
            "shift_hold": evaluate_shift_hold(events["sh"], p_now),
            "predict_shift": evaluate_predict_shift(events["ps_pos"], events["ps_neg"]),
            "backchannel": evaluate_backchannel(events["bc_pos"], events["bc_neg"]),
            "short_long": evaluate_short_long(events["sl"], p_now),
        }

        # Tier 2.5: Calibration
        results["calibration"] = {
            "ece_shift": compute_ece(p_shift_at_events, shift_labels),
            "brier_shift": compute_brier(p_shift_at_events, shift_labels),
        }

        # Tier 3: Latency
        results["latency"] = {
            "eot_latency": compute_eot_latency(events["shifts"], p_shift),
            "fpr_at_thresholds": {
                f"theta_{t}": compute_fpr(events["holds"], p_shift, threshold=t)
                for t in [0.3, 0.4, 0.5, 0.6, 0.7]
            },
            "mst_fpr_curve": compute_mst_fpr_curve(
                events["shifts"], events["holds"], p_shift
            ),
        }

        # Tier 3.5: Efficiency
        results["efficiency"] = benchmark_inference(self.model, ...)

        # Tier 4: Application-level
        results["application"] = {
            "vaqi": compute_vaqi(
                events["shifts"], events["holds"], p_shift
            ),
        }

        # Vietnamese-specific
        results["vietnamese"] = {
            "marker_impact": analyze_marker_impact(...),
            # "dialect_performance": per_dialect_eval(...),  # if available
            # "tonal_analysis": tonal_f0_analysis(...),      # if available
        }

        # Statistical tests
        results["statistical"] = self._compute_significance(results)

        return results
```

### 8.2 Report Output

```
┌──────────────────────────────────────────────────────────────────┐
│  MM-VAP-VI EVALUATION REPORT                                      │
│  Test set: XX conversations, YY hours, ZZ,ZZZ frames             │
│                                                                   │
│  ══════════════════════════════════════════════════               │
│  TIER 1: FRAME-LEVEL                                              │
│  ──────────────────                                               │
│  Cross-Entropy:    X.XX                                           │
│  Perplexity:       X.X                                            │
│  Top-1 Accuracy:   XX.X%                                          │
│  Top-5 Accuracy:   XX.X%                                          │
│  Weighted F1:      0.XXX                                          │
│  Macro F1:         0.XXX                                          │
│                                                                   │
│  ══════════════════════════════════════════════════               │
│  TIER 2: EVENT-BASED (PRIMARY)                                    │
│  ──────────────────────────────                                   │
│  Shift/Hold BA:    XX.X% ± X.X% (95% CI)     ← main metric      │
│    Shift Recall:   XX.X%                                          │
│    Hold Recall:    XX.X%                                          │
│  BC F1:            XX.X% (P=XX.X%, R=XX.X%)  ← secondary        │
│  Short/Long BA:    XX.X%                                          │
│  Predict-Shift:    AUC = 0.XXX                                    │
│                                                                   │
│  Calibration:                                                     │
│    ECE (shift):    0.XXX                                          │
│    Brier (shift):  0.XXX                                          │
│                                                                   │
│  ══════════════════════════════════════════════════               │
│  TIER 3: LATENCY                                                  │
│  ────────────────                                                 │
│  EOT Latency:                                                     │
│    Median:         XXXms                                          │
│    P95:            XXXms                                          │
│    Early Det Rate: XX.X%                                          │
│  FPR @ θ=0.5:     X.X%                                            │
│  MST @ FPR=5%:    XXXms                                           │
│  AU-MFC:           X.XX                                           │
│                                                                   │
│  Efficiency:                                                      │
│    Params:         XXM (trainable: X.XM)                          │
│    Inference:      X.Xms/frame (GPU), XX.Xms/frame (CPU)        │
│    RTF:            0.XX (GPU), 0.XX (CPU)                         │
│                                                                   │
│  ══════════════════════════════════════════════════               │
│  TIER 4: APPLICATION                                              │
│  ────────────────────                                             │
│  VAQI Score:       XX.X / 100                                     │
│    Interruptions:  X.X%                                           │
│    Missed:         X.X%                                           │
│    Latency Score:  0.XXX                                          │
│                                                                   │
│  ══════════════════════════════════════════════════               │
│  VIETNAMESE-SPECIFIC                                              │
│  ────────────────────                                             │
│  Marker benefit:   +X.X% BA khi có discourse marker              │
│  Top markers:      nhé (+X%), mà (+X%), ừ (+X%)                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. Ablation Evaluation Protocol

### 9.1 Ablation nên report metrics nào?

| Ablation | Primary Metric | Secondary | Reason |
|----------|---------------|-----------|--------|
| A1: Acoustic encoder | S/H BA | Perplexity | So sánh SSL vs hand-crafted |
| A2: Modality | S/H BA + BC F1 | MST@5% | BC F1 sensitive to text modality |
| A3: Fusion | S/H BA | Params, Latency | Chất lượng vs efficiency |
| A4: Linguistic features | BC F1 | Marker benefit | HuTu contribution |
| A5: Projection window | S/H BA + Latency | AU-MFC | Window ảnh hưởng cả accuracy và timing |
| A6: Vietnamese analysis | Per-dialect BA | Tonal confusion | Language-specific insights |
| A7: Data scale | S/H BA (learning curve) | — | Data efficiency |

### 9.2 Report Format cho Ablation

```
Table X: Modality ablation (A2)
──────────────────────────────────────────────────────
Model Variant         S/H BA    BC F1    MST@5%   VAQI
──────────────────────────────────────────────────────
Audio-only            78.2%     45.1%    320ms    68.3
Text-only             72.1%     35.4%    N/A      N/A
Audio + PhoBERT       80.8%     49.2%    280ms    73.1
Audio + PhoBERT + HuTu 82.3%   52.0%    260ms    75.8  ← full model
──────────────────────────────────────────────────────

* Bold = best per column. Underline = second best.
  p-values from paired permutation test (n=10000).
  † = significant at p<0.05, ‡ = significant at p<0.01
```

---

## 10. References cho Evaluation Methods

| # | Method | Source | Year |
|---|--------|--------|------|
| 1 | VAP event extraction (S/H, BC, S/L) | Ekstedt & Skantze, Interspeech | 2022 |
| 2 | Multilingual VAP BA | Ekstedt et al., LREC-COLING | 2024 |
| 3 | MST vs FPR curves | Krisp Blog | 2024 |
| 4 | VAQI composite score | Deepgram | 2025 |
| 5 | Sequence-based EoT evaluation | Deepgram | 2025 |
| 6 | FPR @ fixed TPR | LiveKit | 2025 |
| 7 | IoU + F1 for real-time segmentation | SpeculativeETD (arXiv:2503.23439) | 2025 |
| 8 | ECE / Brier Score | Standard calibration literature | — |
| 9 | Bootstrap CI | Efron & Tibshirani | 1993 |
| 10 | Paired permutation test | Noreen (1989) | — |
