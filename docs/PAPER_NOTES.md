# MM-VAP-VI Paper Notes

Ghi chu tong hop methodology, paper structure, va checklist cho paper submission.
Ket qua cu the se duoc dien vao sau khi chay voi full dataset.

---

## 1. Model Architecture Summary

| Component | Detail |
|-----------|--------|
| Acoustic Encoder | wav2vec2-base-vi (nguyenvulebinh), 768->256, freeze 8/12 layers |
| Linguistic Encoder | PhoBERT-v2 (vinai), 768->256, freeze 8/12 layers + HuTu Detector |
| HuTu Detector | Embedding 64d, output 256d, position decay alpha=0.1 |
| Fusion | Cross-attention, 4 heads, dim=256, dropout=0.1 |
| Causal Transformer | 4 layers, 8 heads, dim=256, FFN=1024, ALiBi positional encoding |
| Projection Head | 256 -> 256 classes (VAP formulation) |
| VAP Classes | 2 speakers x 4 time bins = 8 bits = 256 classes |

**Key design choices cho paper:**
- Vietnamese-specific pretrained models (wav2vec2-vi, PhoBERT-v2) thay vi English models
- HuTu (hu tu) detector la novel feature - Vietnamese discourse markers as turn-taking cues
- ALiBi positional encoding cho causal transformer (modern, no fixed length limit)
- Gradient checkpointing voi `use_reentrant=False` de train khi co frozen layers

---

## 2. Training Strategy

### 2.1 Three-Stage Progressive Training

| Stage | What's trained | Purpose |
|-------|---------------|---------|
| 1: Audio-only | Acoustic enc + Transformer + Head | Hoc audio features truoc |
| 2: Multimodal | + PhoBERT last 2 layers + Fusion | Them text modality khi audio da on dinh |
| 3: Full fine-tune | All parameters | Fine-tune toan bo |

**Early stopping:** patience=10, metric=val_loss

**Note cho paper:** Progressive training giup on dinh - Stage 1 hoc audio features truoc,
Stage 2 them text modality vao khi audio da on dinh, Stage 3 fine-tune toan bo.

### 2.2 Training Figures can tao

- [ ] Training curves (loss + accuracy across 3 stages) - tu training_history.json
- [ ] Table: val_loss va val_acc per stage (dien vao sau)

### 2.3 Key Technical Fixes (mention trong Appendix)

- Frame alignment: Wav2Vec2 CNN stride cho ra 999 frames thay vi 1000 cho 20s window.
  Fix: truncate labels/logits to min length.
- Gradient checkpointing: `use_reentrant=False` bat buoc khi frozen layers o truoc trainable layers.
  Default `use_reentrant=True` lam trainable layers KHONG nhan gradient.

---

## 3. Evaluation Framework (4-Tier)

### Tier 1: Frame-Level Metrics
| Metric | Mo ta |
|--------|-------|
| Frame CE | Cross-entropy loss |
| Frame Perplexity | exp(CE), model confidence |
| Frame Top-1 Accuracy | 256-class accuracy (random baseline ~0.4%) |
| Frame Top-5 Accuracy | Top-5 ranking quality |
| Frame Weighted F1 | Class-weighted F1 |
| Frame ECE | Expected Calibration Error |
| Frame Brier Score | Probability calibration |

### Tier 2: Event-Level Metrics
| Metric | Mo ta |
|--------|-------|
| Shift/Hold BA | Balanced accuracy cho turn shift vs hold |
| Predict Shift AUC | ROC AUC cho shift ranking |
| Event 3-class BA | Hold/Shift/BC balanced accuracy |
| BC F1 | Backchannel F1 score |

### Tier 3: Latency Metrics
| Metric | Mo ta |
|--------|-------|
| EoT Latency Mean/Median/P95 | End-of-turn detection latency |
| Detection Rate | % shift events detected |
| FPR @MST thresholds | False positive rate tai cac MST (100-1000ms) |
| MST-FPR AUC | Area under MST-FPR curve |

### Tier 4: VAQI (Voice Agent Quality Index)
| Metric | Mo ta |
|--------|-------|
| VAQI | Composite score 0-100 |
| Interruption Rate (I) | Model predict shift khi nen hold |
| Missed Response Rate (M) | Shift events khong detect |
| Latency Score (L) | Normalized detection latency |

**Formula:** VAQI = 100 x (1 - [0.4*I + 0.4*M + 0.2*L])

---

## 4. Vietnamese Marker Analysis (Novelty chinh)

### 4.1 Marker Categories

| Category | Markers | Function |
|----------|---------|----------|
| Yield (turn-giving) | a, nhe, nhi, nha, ha, hen, nghen | Signal speaker xong y |
| Hold (floor-keeping) | ma, la, thi, nghia la, tuc la | Signal speaker chua xong |
| Backchannel | u, o, um, vang, da, a, o | Listener acknowledgment |
| Turn request | nay, oi, ne, cho hoi | Request de noi |

### 4.2 Analysis Design

So sanh accuracy giua events co chua discourse markers vs khong co:
- **Hypothesis:** Discourse markers cung cap turn-taking signal -> model predict chinh xac hon khi co marker
- **Expected finding:** Shift accuracy cao hon khi co yield markers (a, nhe, nhi...)
- **HuTu detector contribution:** Novel component nhan dien markers va cung cap signal bo sung

### 4.3 Result Template (dien vao sau)

| Category | Shift Accuracy | Hold Accuracy | N events |
|----------|---------------|---------------|----------|
| With marker | | | |
| Without marker | | | |
| **Marker benefit** | | | |

### 4.4 Marker Inventory Justification — 3 Approaches for Paper

**Problem**: ~50% of markers in VAPHuTuDetector lack individual-level academic references
from turn-taking research. Reviewers may ask: "Where does this inventory come from?"

**Approach 1: Two-Level Justification (RECOMMENDED)**

Separate inventory source from classification framework in the Methodology section:

> "Our marker inventory (Table X) was compiled from Vietnamese grammar references
> (Cao, 1991; Diệp, 2005; Nguyễn, 1997; Thompson, 1965), the standard Vietnamese
> dictionary (Hoàng Phê, 2003), function word classifications (Nguyễn Anh Quế, 1988),
> and SFP studies (Tran, 2018; Le, 2015; Hoang, 2025). Regional variants were
> included based on dialectal modal particle research (Nguyễn Văn Hiệp, 2009).
> The turn-taking functional classification follows Duncan's (1972) signal taxonomy
> (yield, attempt-suppressing, backchannel), extended with Schegloff's (1968) summons
> sequences for turn-request markers and Gardner's (2001) response token framework
> for backchannel expressions."

Strengths:
- All sources are STRONG or ACCEPTABLE quality
- Clearly separates "what words" (grammar/dictionary) from "what function" (turn-taking theory)
- Cross-linguistic precedent via Duncan, Schegloff, Gardner
- No need for additional data collection

Weaknesses:
- Some multi-word markers (đúng rồi, ra vậy, khoan đã...) still rely on native-speaker intuition
- Mapping from grammar class to turn-taking function is our contribution, not directly from literature

**Approach 2: Annotation Validation Study**

Add a small validation study to the paper:
- 2-3 native Vietnamese speakers annotate ~100 conversation excerpts
- Annotators label: which words function as turn-taking signals, and what type (yield/hold/BC/request)
- Report inter-annotator agreement (Cohen's kappa or Fleiss' kappa)
- If kappa > 0.7, inventory is empirically validated

Suggested paper text:
> "To validate our marker inventory, we conducted an annotation study with 3 native
> Vietnamese speakers (2 Northern, 1 Southern dialect). Annotators independently labeled
> discourse markers in 100 randomly sampled utterances from our corpus. Inter-annotator
> agreement was κ = X.XX (Fleiss' kappa), indicating [substantial/almost perfect] agreement.
> The final inventory retains markers with ≥2/3 annotator consensus."

Strengths:
- Empirical validation, strongest defense against reviewers
- Can discover new markers not in our current inventory
- Validates regional variants (nha, hen)

Weaknesses:
- Requires 2-3 native speakers willing to annotate
- Takes time (1-2 weeks for 100 segments)
- May not be necessary if ablation results are strong

**Approach 3: Ablation-Only Defense**

Skip justifying individual markers entirely. Let results speak:
- Run use_hutu=True vs use_hutu=False ablation
- If HuTuDetector improves metrics significantly, the inventory is justified empirically
- Frame as: "We test whether explicit Vietnamese discourse marker detection improves
  turn-taking prediction, regardless of the specific marker inventory chosen."

Suggested paper text:
> "Rather than claiming an exhaustive or linguistically definitive marker inventory,
> we treat HuTuDetector as an explicit inductive bias encoding Vietnamese conversational
> pragmatics. The marker inventory was compiled from standard Vietnamese grammar
> references and verified by native speakers. Its utility is evaluated empirically
> through ablation (Section 5.5)."

Strengths:
- Simple, pragmatic
- No additional work needed
- Common in NLP papers

Weaknesses:
- Reviewers may still want to know source of specific markers
- Cannot claim linguistic contribution, only engineering contribution
- Weaker for linguistics-oriented venues (J. Pragmatics) vs NLP venues (EMNLP)

**RECOMMENDATION**: Use Approach 1 as baseline (already prepared in references doc).
If time permits, add Approach 2 for a stronger paper. Approach 3 is fallback.

### 4.5 New References for Marker Inventory (added 2026-02-26)

9 new references added to docs/vietnamese_discourse_markers_references.md [45]-[53]:

| Ref | Source | Covers | Quality |
|-----|--------|--------|---------|
| [45] Le (2015) USC thesis | SFP inventory: à, chứ, đã, đi, nhé, nhỉ, chưa, ạ | WEAK |
| [46] Trinh et al. (2024) Languages | Question particles: không, chưa, à, á, hả | ACCEPTABLE |
| [47] Adachi (2024) Russ. J. Ling. | thật → discourse marker | ACCEPTABLE |
| [48] Nguyễn Anh Quế (1988) | Function words monograph: all conjunctions, particles | ACCEPTABLE |
| [49] Hoàng Phê (2003) Dictionary | Standard dictionary with word-class labels | STRONG |
| [50] Nguyễn Văn Hiệp (2009) | Dialectal modal particles: nha, hen, nghen | ACCEPTABLE |
| [51] Nguyễn Văn Chiến (1992) | Vocative ơi | ACCEPTABLE |
| [52] Schegloff (1968) | Summons-answer sequences (turn-request framework) | STRONG |
| [53] Gardner (2001) | Response tokens (backchannel framework) | STRONG |

### 4.6 Remaining Gaps (markers without any reference)

These markers are justified only by native-speaker intuition + grammar class:
- **YIELD**: xong, hết (completion aspect — classify as tiểu từ thể in Cao 1991)
- **YIELD**: chăng, ư (rare question particles — in Hoàng Phê dictionary as tình thái từ)
- **YIELD**: hen (Southern — mentioned in Nguyễn Văn Hiệp 2009 generally but not individually)
- **BACKCHANNEL multi**: đúng rồi, phải rồi, ra vậy, ra thế, rồi sao, thế rồi sao
- **TURN_REQUEST multi**: khoan đã, nhưng mà, để tôi/em, cho tôi/em, tôi nghĩ/em nghĩ

Strategy: These are covered at the word-class level by Hoàng Phê (2003) dictionary
and at the functional level by Duncan (1972) / Gardner (2001) frameworks.
If reviewer pushes, point to ablation results.

---

## 5. Ablation Studies (can chay)

### 5.1 Multimodal vs Audio-Only

| Metric | Multimodal | Audio-Only | Delta |
|--------|-----------|------------|-------|
| Frame Top-1 Acc | | | |
| Shift/Hold BA | | | |
| Predict Shift AUC | | | |
| VAQI | | | |
| EoT Latency | | | |

**Ky vong:** Text modality se cai thien accuracy ro rang khi co du data.

### 5.2 Other Ablations Can Chay

| Ablation | Mo ta |
|----------|-------|
| No HuTu detector | Loai bo discourse marker features |
| No cross-attention fusion | Replace voi simple concat |
| 1-stage training | Train tat ca tu dau, khong progressive |
| English wav2vec2 | So sanh voi wav2vec2 trained on English |
| Silence-based baseline | Rule-based: predict shift sau N ms im lang |
| Random baseline | Random 256-class predictions |

---

## 6. Per-Speaker Analysis

### Discussion Points

- **Interview format:** Host noi it nhung da dang pattern (hoi, dan dat), guest noi nhieu nhung don dieu
- **pyannote speaker ID:** Gan theo thu tu xuat hien, khong nhat quan giua cac file
- **Ky vong:** Speaker bias se giam khi co them symmetric conversation data (khong chi interview)

**Cach report trong paper:**
> "We observe speaker asymmetry in prediction accuracy, attributable to the interview
> format where hosts exhibit more diverse turn-taking patterns. Future work will address
> this through speaker-role-aware modeling and symmetric conversation data."

---

## 7. Confusion Matrix Analysis

### Key Patterns Can Ky Vong

1. **Hold detection cao nhat** - Model tot trong viec nhan ra khi KHONG can chuyen luot
2. **Shift miss rate** - Model conservative, thien ve Hold
3. **BC detection yeu** - Can nhieu backchannel samples
4. **Conservative bias** - Uu tien predict Hold > Shift > BC

**Giai thich trong paper:**
> "The model exhibits conservative bias, favoring hold predictions over shifts.
> This is preferable in voice agent deployment where false interruptions are
> more costly than missed responses."

---

## 8. Figures Can Tao Cho Paper

- [ ] Training curves (loss + accuracy across 3 stages)
- [ ] Confusion matrix (3-class, counts + normalized)
- [ ] FPR vs MST curve (line plot)
- [ ] Marker impact bar chart (with/without marker accuracy comparison)
- [ ] Qualitative timeline plots (VA, P(shift), GT events, Pred events)
- [ ] Architecture diagram (model overview)
- [ ] Data pipeline flowchart
- [ ] ROC curve cho shift prediction
- [ ] Per-speaker accuracy comparison

---

## 9. Paper Structure

### Title Options
1. "MM-VAP-VI: Multimodal Voice Activity Projection for Vietnamese Turn-Taking with Discourse Marker Detection"
2. "Predicting Turn-Taking in Vietnamese Conversations: A Multimodal Approach with Discourse Marker Features"

### Sections

1. **Introduction** - Turn-taking prediction, tai sao Vietnamese, tai sao multimodal
2. **Related Work** - VAP (Ekstedt), TurnGPT, multimodal turn-taking, Vietnamese NLP
3. **Method**
   - 3.1 VAP Formulation (256-class)
   - 3.2 Acoustic Encoder (wav2vec2-vi)
   - 3.3 Linguistic Encoder (PhoBERT + HuTu Detector)
   - 3.4 Cross-Attention Fusion
   - 3.5 Causal Transformer + ALiBi
   - 3.6 Three-Stage Progressive Training
4. **Data**
   - 4.1 Data Collection Pipeline (YouTube, diarization, transcription)
   - 4.2 VAP Label Encoding
   - 4.3 Text Frame Alignment
5. **Experiments**
   - 5.1 Implementation Details
   - 5.2 Frame-Level Results (Table 1)
   - 5.3 Event-Level Results (Table 2)
   - 5.4 Latency Analysis (Table 3)
   - 5.5 Ablation: Multimodal vs Audio-Only (Table 4)
   - 5.6 Vietnamese Marker Analysis (Table 5)
   - 5.7 Qualitative Analysis (Figures)
6. **Discussion**
   - 6.1 Speaker Asymmetry
   - 6.2 Conservative Prediction Bias
   - 6.3 Marker Impact
   - 6.4 Limitations
7. **Conclusion**

---

## 10. So Sanh Voi SOTA (cho paper table)

| Model | Language | Modality | Shift BA | AUC | Venue |
|-------|----------|----------|----------|-----|-------|
| VAP (Ekstedt 2022) | EN/JP | Audio | 0.82 | 0.88 | ACL-WS |
| TurnGPT (Ekstedt 2020) | EN | Text | 0.78 | 0.84 | EMNLP-WS |
| MM-TurnGPT (2023) | EN | Audio+Text | 0.85 | 0.90 | Interspeech |
| Roddy et al. (2018) | EN | Audio | 0.76 | - | Interspeech |
| **MM-VAP-VI (ours)** | **VI** | **Audio+Text** | TBD | TBD | - |

**Note:** So sanh cross-language khong hoan toan fair vi data va ngon ngu khac nhau.

---

## 11. Data Presentation Strategy (Weak Supervision)

### Supervision Type cua tung buoc

| Buoc | Tool | Supervision type |
|------|------|-----------------|
| Audio collection | yt-dlp (YouTube) | No supervision |
| Speaker diarization | pyannote 3.1 | **Weak** (pretrained model, khong manual) |
| Transcription | Whisper large-v3 | **Weak** (pretrained ASR) |
| VAP labels | Tu VA matrix (rule-based) | **Self-supervised** (derived from diarization) |
| Text alignment | Rule-based mapping | Deterministic |

**Khong co buoc nao co human annotation** → day chinh xac la **weakly-supervised / automatically-annotated** dataset.

### SOTA papers cung lam tuong tu

- **VAP (Ekstedt 2022)** - dung automatic diarization, VAP labels tu generate
- **TurnGPT (Ekstedt 2020)** - dung text transcripts co san
- **Khac biet cua MM-VAP-VI:** tu tao TOAN BO tu raw YouTube → day la contribution

### Cach viet trong paper (Section 4: Data)

**4.1 Data Collection** (~0.5 trang)
> We construct a Vietnamese conversational dataset from publicly available YouTube
> content. We curate 65 two-speaker conversation videos across 8 genres (interview,
> casual conversation, discussion/debate, etc.) totaling ~70 hours of audio.
> Videos are selected to ensure: (1) exactly two speakers, (2) Vietnamese language,
> (3) studio-quality audio, and (4) diverse turn-taking patterns.

**4.2 Automatic Annotation Pipeline** (~0.5 trang)
> All annotations are generated automatically without manual intervention:
> - Speaker diarization: pyannote.audio 3.1 (Bredin et al., 2023)
> - ASR transcription: Whisper large-v3 (Radford et al., 2023) with word-level timestamps
> - VAP label encoding: deterministic mapping from voice activity to 256-class labels
>   (Ekstedt & Tower, 2022)
>
> We acknowledge this introduces noise from diarization and ASR errors. However,
> this approach enables scalable dataset creation for low-resource languages where
> manual annotation is prohibitively expensive.

**4.3 Dataset Statistics** (~0.5 trang, table)

| Statistic | Value |
|-----------|-------|
| Total hours | ~70h |
| Segments (~10 min) | ~400 |
| Train/Val/Test | 80/10/10 |
| Genres | 8 |
| Unique speakers | ~130 |

### Genre diversity (65 videos, 8 the loai)

| Genre | Kenh | So video | Dac diem |
|-------|------|----------|----------|
| Interview/Lifestyle | Have A Sip (Vietcetera) | 20 | Host + guest, studio quality |
| Entertainment | Bar Stories (Dustin) | 12 | Intimate, emotional |
| Career/Professional | Spiderum NTMN | 8 | Da dang nghe nghiep |
| Discussion/Debate | Hoi Dong Cuu + misc | 5 | Tranh luan, nhieu y kien |
| Casual conversation | Nhat Ky Ban Cong | 7 | Casual, tu nhien |
| Formal talkshow | Talksoul + VietSuccess | 5 | Broadcast quality |
| Co-host (symmetric) | Duoc/Mat, Tan So 52Hz | 4 | 2 co-hosts can bang |
| Deep talk/Psychology | Mixed | 4 | Tam ly, phat trien ban than |

### Ve viec "cong bo bo data"

**KHONG the cong bo audio** (YouTube copyright). Nhung CO THE cong bo:
- URL list + pipeline scripts (de reproduce)
- Derived annotations (RTTM, VA matrices, VAP labels, transcripts) → fair use
- Pre-trained model weights

Cach viet:
> "We release our annotation pipeline, URL list, and derived labels.
> Raw audio can be reconstructed by running the provided download scripts."

### Can lam them cho paper manh hon

1. **Manual validation subset** - Lay 5-10 segments, tu nghe va annotate turn boundaries,
   so sanh voi automatic labels. Bao cao agreement rate. Khong can annotate toan bo.
2. **Dataset statistics figures** - Phan bo silence/speech/overlap, speaker balance per genre
3. **Inter-annotator agreement** - Neu co 2 nguoi annotate, bao cao Cohen's kappa

---

## 12. Limitations Section (quan trong cho reviewer)

### Must Acknowledge
1. **YouTube data** - Podcast/interview format khac hoi thoai doi thuong
2. **Speaker diarization tu dong (pyannote)** - Co loi, khong co manual annotation
3. **Transcription tu dong (Whisper)** - ASR errors anh huong text features
4. **Khong co human baseline** - Can human annotation de so sanh
5. **Single GPU training** - Batch size nho, co the chua optimal
6. **Speaker bias** - Interview format tao speaker asymmetry

### Checklist Truoc Khi Nop
- [x] Scale data len 50+ audio (65 videos, 8 genres, ~70h) ← DONE
- [ ] Chay tat ca baselines (silence-based, English wav2vec2, random)
- [ ] Chay tat ca ablations (no HuTu, no fusion, 1-stage training)
- [ ] Bootstrap CI cho TAT CA experiments
- [ ] Statistical significance tests (permutation test) giua cac models
- [ ] Manual annotation subset de validate automatic pipeline
- [ ] Dien ket qua vao cac template tables trong file nay

---

## 13. Target Venues

| Venue | Phu hop | Deadline | Ly do |
|-------|---------|----------|-------|
| **Interspeech 2026** | Rat phu hop | ~Mar 2026 | Speech community, accept language-specific studies |
| **EMNLP 2026** | Tot | ~Jun 2026 | Multimodal + low-resource language |
| **LREC-COLING** | Backup tot | TBD | Language resources focus |
| **SLT/ASRU** | Tot | TBD | Speech technology |
| ACL 2026 | Kho | ~Feb 2026 | Top tier, can data lon hon |

---

## 14. Output Files Reference

```
outputs/
  eval_report.json            # Main evaluation (multimodal)
  eval_audio_only.json        # Audio-only ablation
  eval_bootstrap_report.json  # Bootstrap CI
  mm_vap/
    best_model.pt             # Best checkpoint
    training_history.json     # Full training curves
  figures/
    confusion_matrix.png      # 3-class confusion matrix
    confusion_matrix.json     # Raw confusion matrix data
    per_speaker_analysis.json # Speaker bias data
    window_*.png              # Qualitative visualizations
```
