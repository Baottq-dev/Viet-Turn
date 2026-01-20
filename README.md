# 🇻🇳 Viet-TurnEdge

**Vietnamese Turn-Taking Prediction using Multimodal Deep Learning**

A research project for predicting turn-taking (turn yield, hold, backchannel) in Vietnamese conversations using a hybrid acoustic-linguistic model.

## 🏗️ Architecture

```
Audio → [Mel + F0 + Energy] → Causal TCN → ┐
                                            ├→ GMU Fusion → Classifier → Prediction
Text  → [PhoBERT + Hư từ] → Linear     → ┘
```

**Components:**
- **Acoustic Branch:** Causal Dilated TCN (4 layers, ~300ms receptive field)
- **Linguistic Branch:** PhoBERT-base-v2 with Vietnamese discourse marker detection
- **Fusion:** Gated Multimodal Unit (GMU) with learned modality weighting
- **Output:** 3 classes - Turn-Yield, Turn-Hold, Backchannel

## 📁 Project Structure

```
Viet-Turn/
├── configs/           # YAML configurations
├── data/              # Datasets (raw, processed, labels)
├── src/
│   ├── data/          # Audio processing, labeling
│   ├── models/        # TCN, PhoBERT, GMU, full model
│   ├── training/      # Trainer, losses, metrics
│   └── asr/           # PhoWhisper integration
├── scripts/           # Training, evaluation scripts
├── tests/             # Unit tests
└── notebooks/         # Experiments
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Model

```python
from src.models import VietTurnEdge
import torch

model = VietTurnEdge()
audio_features = torch.randn(1, 42, 100)  # (B, features, T)

output = model(audio_features)
print(output['probs'].shape)  # (1, 100, 3)
```

## 📊 Technology Stack

| Component | Choice |
|-----------|--------|
| ASR | PhoWhisper-base |
| Linguistic | PhoBERT-base-v2 |
| Loss | Focal Loss (γ=2.0) |
| LLM Labeling | Gemini API |

## 📄 License

MIT License
