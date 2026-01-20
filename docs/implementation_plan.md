# 🔬 Viet-TurnEdge Research Plan

> **Project:** Turn-Taking Prediction System for Vietnamese  
> **Type:** Research / Prototype  
> **Architecture:** Late Fusion Hybrid (TCN + PhoBERT + GMU)

---

## 📋 Executive Summary

Hệ thống dự báo chuyển lượt (turn-taking) cho tiếng Việt, sử dụng kiến trúc multimodal:
- **Acoustic Branch:** Causal Dilated TCN xử lý audio features
- **Linguistic Branch:** PhoBERT-base-v2 xử lý text từ ASR  
- **Fusion Layer:** Gated Multimodal Unit (GMU) hợp nhất hai luồng
- **Output:** 3 classes - Turn-Yield, Turn-Hold, Backchannel

### ✅ Confirmed Technology Stack

| Component | Choice | Reasoning |
|-----------|--------|----------|
| **ASR** | `vinai/PhoWhisper-base` | SOTA accuracy (~5-8% WER), sliding window for pseudo-streaming |
| **Linguistic** | `vinai/phobert-base-v2` | Full model for best semantic understanding of hư từ |
| **LLM Labeling** | Gemini API | Cost-effective, good Vietnamese support |

---

## 🏗️ Project Structure

```
Viet-Turn/
├── 📂 configs/                    # Configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── inference_config.yaml
│
├── 📂 data/                       # Data directory
│   ├── raw/                       # Raw downloaded data
│   │   ├── vlsp2020/
│   │   └── bud500/
│   ├── processed/                 # Processed features
│   │   ├── audio_features/
│   │   └── text_features/
│   └── labels/                    # Turn-taking labels
│
├── 📂 src/                        # Source code
│   ├── 📂 data/                   # Data processing
│   │   ├── __init__.py
│   │   ├── audio_processor.py     # Mel, F0, Energy extraction
│   │   ├── text_processor.py      # Tokenization, hư từ detection
│   │   ├── dataset.py             # PyTorch Dataset classes
│   │   └── labeling/              # LLM-based labeling
│   │       ├── diarization.py
│   │       ├── llm_judge.py
│   │       └── label_pipeline.py
│   │
│   ├── 📂 models/                 # Model architectures
│   │   ├── __init__.py
│   │   ├── acoustic_branch.py     # Causal Dilated TCN
│   │   ├── linguistic_branch.py   # TinyPhoBERT
│   │   ├── fusion.py              # GMU Layer
│   │   ├── classifier.py          # Final FC + Softmax
│   │   └── viet_turn_edge.py      # Full model
│   │
│   ├── 📂 asr/                    # ASR integration
│   │   ├── __init__.py
│   │   ├── vosk_stream.py         # Vosk streaming wrapper
│   │   ├── phowhisper_async.py    # PhoWhisper async inference
│   │   └── asr_manager.py         # Hybrid ASR manager
│   │
│   ├── 📂 training/               # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py             # Main training loop
│   │   ├── losses.py              # Focal Loss implementation
│   │   └── metrics.py             # TRP F1, Latency metrics
│   │
│   └── 📂 inference/              # Inference & deployment
│       ├── __init__.py
│       ├── realtime_pipeline.py   # Real-time inference
│       ├── onnx_export.py         # ONNX conversion
│       └── quantize.py            # INT8 quantization
│
├── 📂 scripts/                    # Utility scripts
│   ├── download_data.py
│   ├── preprocess_audio.py
│   ├── generate_labels.py
│   ├── train.py
│   ├── evaluate.py
│   └── demo.py
│
├── 📂 tests/                      # Unit tests
│   ├── test_audio_processor.py
│   ├── test_models.py
│   ├── test_asr.py
│   └── test_pipeline.py
│
├── 📂 notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_experiments.ipynb
│
├── 📂 deploy/                     # Deployment configs
│   ├── raspberry_pi/
│   │   ├── setup.sh
│   │   └── run_demo.py
│   └── docker/
│       └── Dockerfile
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 📦 Phase 1: Foundation (Week 1)

### 1.1 Project Setup

#### [NEW] [requirements.txt](file:///f:/Viet-Turn/requirements.txt)

```txt
# Core
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Audio Processing
librosa>=0.10.0
pyaudio>=0.2.13
soundfile>=0.12.0
praat-parselmouth>=0.4.3  # F0 extraction

# ASR
vosk>=0.3.45
transformers>=4.36.0
openai-whisper>=20231117
onnxruntime>=1.16.0

# NLP
underthesea>=6.8.0  # Vietnamese NLP
pyvi>=0.1.1

# Training
wandb>=0.16.0
tensorboard>=2.15.0

# LLM Labeling
google-generativeai>=0.3.0
anthropic>=0.8.0

# Utilities
pyyaml>=6.0.0
tqdm>=4.66.0
pandas>=2.1.0
matplotlib>=3.8.0
seaborn>=0.13.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

#### [NEW] [configs/model_config.yaml](file:///f:/Viet-Turn/configs/model_config.yaml)

```yaml
# Viet-TurnEdge Model Configuration

model:
  name: "viet_turn_edge"
  num_classes: 3  # Turn-Yield, Turn-Hold, Backchannel

acoustic_branch:
  type: "causal_dilated_tcn"
  input_dim: 42  # 40 Mel + F0 + Energy
  hidden_dim: 128
  output_dim: 64
  num_layers: 4
  kernel_size: 3
  dilation_base: 2  # [1, 2, 4, 8]
  dropout: 0.1

linguistic_branch:
  type: "phobert_base"
  pretrained: "vinai/phobert-base-v2"  # Full model, not distilled
  hidden_dim: 768
  output_dim: 64
  freeze_embeddings: true  # Freeze for efficiency

asr:
  model: "vinai/PhoWhisper-base"  # Or PhoWhisper-small
  sliding_window_sec: 3.0  # Pseudo-streaming window
  stride_sec: 0.5  # Overlap stride

fusion:
  type: "gmu"
  acoustic_dim: 64
  linguistic_dim: 64
  hidden_dim: 128
  output_dim: 64

classifier:
  hidden_dim: 32
  num_classes: 3
  dropout: 0.2

audio:
  sample_rate: 16000
  frame_length_ms: 20
  frame_shift_ms: 10
  n_mels: 40
  n_fft: 512
  use_f0: true
  use_energy: true
```

---

## 📂 Phase 2: Data Pipeline (Week 2-3)

### 2.1 Data Download & Preparation

#### [NEW] [scripts/download_data.py](file:///f:/Viet-Turn/scripts/download_data.py)

Download datasets từ các nguồn đã research:
- VLSP2020-100h từ Hugging Face
- Setup directory structure

### 2.2 Audio Feature Extraction

#### [NEW] [src/data/audio_processor.py](file:///f:/Viet-Turn/src/data/audio_processor.py)

```python
"""
Audio feature extraction for Vietnamese turn-taking.
Features: Log-Mel Spectrogram + F0 Pitch + Energy
"""

import torch
import librosa
import numpy as np
import parselmouth
from typing import Tuple, Optional

class AudioProcessor:
    """Extract acoustic features optimized for Vietnamese prosody."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_ms: int = 20,
        frame_shift_ms: int = 10,
        n_mels: int = 40,
        n_fft: int = 512,
        f_min: float = 50.0,
        f_max: float = 400.0,  # Vietnamese F0 range
    ):
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * frame_length_ms / 1000)
        self.frame_shift = int(sample_rate * frame_shift_ms / 1000)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        
    def extract_mel(self, audio: np.ndarray) -> np.ndarray:
        """Extract log-mel spectrogram."""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.frame_shift,
            win_length=self.frame_length,
            n_mels=self.n_mels,
            fmin=80,
            fmax=self.sample_rate // 2
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel  # Shape: (n_mels, time)
    
    def extract_f0(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 contour using Praat/Parselmouth."""
        snd = parselmouth.Sound(audio, self.sample_rate)
        pitch = snd.to_pitch(
            time_step=self.frame_shift / self.sample_rate,
            pitch_floor=self.f_min,
            pitch_ceiling=self.f_max
        )
        f0 = pitch.selected_array['frequency']
        f0[f0 == 0] = np.nan  # Unvoiced = NaN
        f0 = np.nan_to_num(f0, nan=0.0)
        return f0  # Shape: (time,)
    
    def extract_energy(self, audio: np.ndarray) -> np.ndarray:
        """Extract frame-level energy (RMS)."""
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.frame_shift
        )[0]
        return energy  # Shape: (time,)
    
    def __call__(self, audio: np.ndarray) -> torch.Tensor:
        """Extract all features and concatenate."""
        mel = self.extract_mel(audio)     # (40, T)
        f0 = self.extract_f0(audio)       # (T,)
        energy = self.extract_energy(audio)  # (T,)
        
        # Align lengths
        min_len = min(mel.shape[1], len(f0), len(energy))
        mel = mel[:, :min_len]
        f0 = f0[:min_len]
        energy = energy[:min_len]
        
        # Normalize
        f0 = (f0 - f0.mean()) / (f0.std() + 1e-8)
        energy = (energy - energy.mean()) / (energy.std() + 1e-8)
        
        # Concatenate: (42, T) = 40 mel + 1 f0 + 1 energy
        features = np.vstack([
            mel,
            f0.reshape(1, -1),
            energy.reshape(1, -1)
        ])
        
        return torch.FloatTensor(features)
```

### 2.3 LLM-based Turn Labeling

#### [NEW] [src/data/labeling/llm_judge.py](file:///f:/Viet-Turn/src/data/labeling/llm_judge.py)

```python
"""
LLM-as-a-Judge for automatic turn-taking label generation.
Uses GPT-4o or Claude to classify Turn-Relevance Points.
"""

import json
from typing import List, Dict, Tuple
import google.generativeai as genai

SYSTEM_PROMPT = """Bạn là chuyên gia ngôn ngữ học về hội thoại tiếng Việt.

Nhiệm vụ: Phân tích đoạn hội thoại và xác định điểm chuyển lượt (Turn-Relevance Points).

Với mỗi câu/phát ngôn, hãy gán nhãn:
- YIELD: Người nói kết thúc, sẵn sàng để người khác nói (có hư từ: nhé, nhỉ, à, hả, ạ)
- HOLD: Người nói chưa xong, sẽ tiếp tục (có hư từ: mà, thì, là, nhưng mà, vì)  
- BACKCHANNEL: Phản hồi ngắn không chiếm lượt (ừ, vâng, ờ, à, thế à, vậy hả)

Trả về JSON array với format:
[
  {"text": "...", "speaker": "A/B", "label": "YIELD/HOLD/BACKCHANNEL", "confidence": 0.0-1.0}
]
"""

class LLMTurnLabeler:
    """Use LLM to automatically label turn-taking points."""
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = genai.GenerativeModel(model)
        
    def label_dialogue(
        self, 
        dialogue: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        Label a dialogue with turn-taking annotations.
        
        Args:
            dialogue: List of {"speaker": "A", "text": "...", "start": 0.0, "end": 1.5}
        
        Returns:
            List of labeled utterances with turn-taking labels
        """
        # Format dialogue for LLM
        formatted = "\n".join([
            f"[{u['speaker']}] ({u['start']:.1f}s - {u['end']:.1f}s): {u['text']}"
            for u in dialogue
        ])
        
        prompt = f"""Phân tích đoạn hội thoại sau và gán nhãn turn-taking:

{formatted}

Trả về JSON array với nhãn cho từng phát ngôn."""

        response = self.model.generate_content(
            [{"role": "user", "parts": [prompt]}],
            generation_config={"response_mime_type": "application/json"}
        )
        
        try:
            labels = json.loads(response.text)
            return self._merge_labels(dialogue, labels)
        except json.JSONDecodeError:
            return self._fallback_labeling(dialogue)
    
    def _merge_labels(
        self, 
        dialogue: List[Dict], 
        labels: List[Dict]
    ) -> List[Dict]:
        """Merge LLM labels back into dialogue structure."""
        for i, (utt, label) in enumerate(zip(dialogue, labels)):
            utt['label'] = label.get('label', 'HOLD')
            utt['confidence'] = label.get('confidence', 0.5)
        return dialogue
    
    def _fallback_labeling(self, dialogue: List[Dict]) -> List[Dict]:
        """Rule-based fallback when LLM fails."""
        yield_markers = ['nhé', 'nhỉ', 'à', 'hả', 'ạ', 'không', 'chứ']
        hold_markers = ['mà', 'thì', 'là', 'nhưng', 'vì', 'nên']
        backchannel = ['ừ', 'vâng', 'ờ', 'thế à', 'vậy hả', 'à']
        
        for utt in dialogue:
            text = utt['text'].lower().strip()
            
            # Check backchannel first (short responses)
            if len(text.split()) <= 3 and any(bc in text for bc in backchannel):
                utt['label'] = 'BACKCHANNEL'
            elif any(text.endswith(ym) for ym in yield_markers):
                utt['label'] = 'YIELD'
            elif any(hm in text for hm in hold_markers):
                utt['label'] = 'HOLD'
            else:
                utt['label'] = 'YIELD'  # Default: end of utterance = yield
            
            utt['confidence'] = 0.6  # Lower confidence for rule-based
        
        return dialogue
```

---

## 🧠 Phase 3: Model Development (Week 4-6)

### 3.1 Acoustic Branch - Causal Dilated TCN

#### [NEW] [src/models/acoustic_branch.py](file:///f:/Viet-Turn/src/models/acoustic_branch.py)

```python
"""
Causal Dilated Temporal Convolutional Network for acoustic features.
Optimized for real-time Vietnamese turn prediction.
"""

import torch
import torch.nn as nn
from typing import List

class CausalConv1d(nn.Module):
    """1D Convolution with causal padding (no future leakage)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        # Remove future padding
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNResidualBlock(nn.Module):
    """Residual block with dilated causal convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        out = self.relu(out + residual)
        return out


class CausalDilatedTCN(nn.Module):
    """
    Causal Dilated TCN for acoustic turn-taking prediction.
    
    Receptive field = sum of (kernel_size - 1) * dilation for each layer
    With 4 layers, kernel=3, dilations=[1,2,4,8]: RF = 2*(1+2+4+8) = 30 frames = ~300ms
    """
    
    def __init__(
        self,
        input_dim: int = 42,  # 40 mel + 1 f0 + 1 energy
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        
        # Build dilated layers
        layers = []
        dilations = [2 ** i for i in range(num_layers)]  # [1, 2, 4, 8]
        
        for i, dilation in enumerate(dilations):
            layers.append(TCNResidualBlock(
                hidden_dim, hidden_dim, kernel_size, dilation, dropout
            ))
        
        self.tcn_layers = nn.ModuleList(layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim, T) - Audio features over time
            
        Returns:
            (B, T, output_dim) - Contextualized acoustic representations
        """
        # Project input
        out = self.input_proj(x)  # (B, hidden_dim, T)
        
        # Apply TCN layers
        for layer in self.tcn_layers:
            out = layer(out)
        
        # Transpose and project output
        out = out.transpose(1, 2)  # (B, T, hidden_dim)
        out = self.output_proj(out)  # (B, T, output_dim)
        
        return out
```

### 3.2 Linguistic Branch - TinyPhoBERT

#### [NEW] [src/models/linguistic_branch.py](file:///f:/Viet-Turn/src/models/linguistic_branch.py)

```python
"""
Distilled PhoBERT for Vietnamese linguistic turn-taking features.
Focuses on detecting hư từ (discourse markers).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Tuple

class HuTuDetector(nn.Module):
    """Lightweight hư từ (discourse marker) detector."""
    
    # Vietnamese discourse markers for turn prediction
    YIELD_MARKERS = {'nhé', 'nhỉ', 'à', 'hả', 'ạ', 'không', 'chứ', 'hen'}
    HOLD_MARKERS = {'mà', 'thì', 'là', 'nhưng', 'vì', 'nên', 'nếu', 'khi'}
    BACKCHANNEL_MARKERS = {'ừ', 'vâng', 'ờ', 'dạ', 'ừm', 'ừ hử'}
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.all_markers = (
            self.YIELD_MARKERS | self.HOLD_MARKERS | self.BACKCHANNEL_MARKERS
        )
        self.marker_to_idx = {m: i for i, m in enumerate(self.all_markers)}
        
        self.marker_embed = nn.Embedding(len(self.all_markers) + 1, embedding_dim)
        
    def forward(self, text: str) -> torch.Tensor:
        """Detect markers and return embedding."""
        tokens = text.lower().split()
        
        marker_ids = []
        for token in tokens:
            if token in self.marker_to_idx:
                marker_ids.append(self.marker_to_idx[token])
        
        if not marker_ids:
            marker_ids = [len(self.all_markers)]  # No marker token
            
        ids_tensor = torch.LongTensor(marker_ids)
        embed = self.marker_embed(ids_tensor).mean(dim=0)
        return embed


class TinyPhoBERT(nn.Module):
    """
    Distilled PhoBERT for turn-taking prediction.
    Uses only first 4 layers + marker detection.
    """
    
    def __init__(
        self,
        pretrained: str = "vinai/phobert-base-v2",
        num_layers: int = 4,
        output_dim: int = 64,
        use_marker_detection: bool = True
    ):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        
        # Load full PhoBERT
        full_model = AutoModel.from_pretrained(pretrained)
        
        # Extract only first N layers (distillation-like)
        self.embeddings = full_model.embeddings
        self.encoder_layers = nn.ModuleList(
            full_model.encoder.layer[:num_layers]
        )
        
        # Freeze embeddings for efficiency
        for param in self.embeddings.parameters():
            param.requires_grad = False
            
        hidden_size = full_model.config.hidden_size  # 768
        
        # Marker detector
        self.use_markers = use_marker_detection
        if use_marker_detection:
            self.marker_detector = HuTuDetector(embedding_dim=32)
            projection_input = hidden_size + 32
        else:
            projection_input = hidden_size
            
        # Output projection
        self.output_proj = nn.Linear(projection_input, output_dim)
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text: Optional[str] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, seq_len) - Tokenized input
            attention_mask: (B, seq_len)
            text: Original text for marker detection
            
        Returns:
            (B, output_dim) - Linguistic representation
        """
        # Embedding
        hidden_states = self.embeddings(input_ids)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        
        # Pool: use [CLS] token
        pooled = hidden_states[:, 0, :]  # (B, hidden_size)
        
        # Add marker features
        if self.use_markers and text is not None:
            marker_embed = self.marker_detector(text)  # (32,)
            marker_embed = marker_embed.unsqueeze(0).expand(pooled.size(0), -1)
            pooled = torch.cat([pooled, marker_embed.to(pooled.device)], dim=-1)
        
        # Project to output dim
        output = self.output_proj(pooled)  # (B, output_dim)
        
        return output
```

### 3.3 GMU Fusion Layer

#### [NEW] [src/models/fusion.py](file:///f:/Viet-Turn/src/models/fusion.py)

```python
"""
Gated Multimodal Unit (GMU) for fusing acoustic and linguistic features.
Learns to dynamically weight each modality based on context.
"""

import torch
import torch.nn as nn

class GatedMultimodalUnit(nn.Module):
    """
    GMU Fusion Layer.
    
    h = z ⊙ tanh(W_a · h_a) + (1-z) ⊙ tanh(W_t · h_t)
    z = σ(W_z · [h_a; h_t])
    
    Where:
        h_a: Acoustic features
        h_t: Linguistic/Text features  
        z: Learned gate (0-1)
    """
    
    def __init__(
        self,
        acoustic_dim: int = 64,
        linguistic_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64
    ):
        super().__init__()
        
        # Modality transformations
        self.acoustic_transform = nn.Linear(acoustic_dim, hidden_dim)
        self.linguistic_transform = nn.Linear(linguistic_dim, hidden_dim)
        
        # Gate computation
        self.gate = nn.Sequential(
            nn.Linear(acoustic_dim + linguistic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        h_acoustic: torch.Tensor,
        h_linguistic: torch.Tensor,
        return_gate: bool = False
    ) -> torch.Tensor:
        """
        Fuse acoustic and linguistic representations.
        
        Args:
            h_acoustic: (B, T, acoustic_dim) or (B, acoustic_dim)
            h_linguistic: (B, linguistic_dim) - Usually pooled from BERT
            return_gate: If True, also return gate values for visualization
            
        Returns:
            Fused representation (B, T, output_dim) or (B, output_dim)
        """
        # Handle different input shapes
        has_time = len(h_acoustic.shape) == 3
        
        if has_time:
            B, T, _ = h_acoustic.shape
            # Expand linguistic to match time dimension
            h_linguistic = h_linguistic.unsqueeze(1).expand(-1, T, -1)
        
        # Compute gate
        concat = torch.cat([h_acoustic, h_linguistic], dim=-1)
        z = self.gate(concat)  # (B, [T,] hidden_dim)
        
        # Transform modalities
        h_a_transformed = torch.tanh(self.acoustic_transform(h_acoustic))
        h_t_transformed = torch.tanh(self.linguistic_transform(h_linguistic))
        
        # Gated fusion
        fused = z * h_a_transformed + (1 - z) * h_t_transformed
        
        # Project to output
        output = self.output_proj(fused)
        
        if return_gate:
            return output, z.mean(dim=-1)  # Return average gate value
        return output
```

### 3.4 Full Model Architecture

#### [NEW] [src/models/viet_turn_edge.py](file:///f:/Viet-Turn/src/models/viet_turn_edge.py)

```python
"""
VietTurnEdge: Full hybrid model for Vietnamese turn-taking prediction.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .acoustic_branch import CausalDilatedTCN
from .linguistic_branch import TinyPhoBERT
from .fusion import GatedMultimodalUnit


class VietTurnEdge(nn.Module):
    """
    Hybrid multimodal model for Vietnamese turn-taking prediction.
    
    Architecture:
        Audio → TCN → GMU → Classifier → Turn Prediction
        Text → TinyBERT ↗
    """
    
    LABELS = ['turn_yield', 'turn_hold', 'backchannel']
    
    def __init__(
        self,
        acoustic_config: dict,
        linguistic_config: dict,
        fusion_config: dict,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Acoustic branch
        self.acoustic_branch = CausalDilatedTCN(**acoustic_config)
        
        # Linguistic branch
        self.linguistic_branch = TinyPhoBERT(**linguistic_config)
        
        # Fusion layer
        self.fusion = GatedMultimodalUnit(**fusion_config)
        
        # Classifier
        fusion_output_dim = fusion_config.get('output_dim', 64)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(
        self,
        audio_features: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            audio_features: (B, features, T) - Mel + F0 + Energy
            input_ids: (B, seq_len) - Tokenized text (optional)
            attention_mask: (B, seq_len)
            text: Raw text for marker detection
            return_features: If True, return intermediate features
            
        Returns:
            Dictionary with:
                - logits: (B, T, num_classes) - Class logits
                - probs: (B, T, num_classes) - Class probabilities
                - gate_values: (B, T) - GMU gate values (0=text, 1=audio)
        """
        # Acoustic branch
        h_acoustic = self.acoustic_branch(audio_features)  # (B, T, acoustic_dim)
        
        # Linguistic branch
        if input_ids is not None:
            h_linguistic = self.linguistic_branch(
                input_ids, attention_mask, text
            )  # (B, linguistic_dim)
        else:
            # No text available - use zero vector
            B = audio_features.size(0)
            h_linguistic = torch.zeros(
                B, self.fusion.acoustic_transform.in_features,
                device=audio_features.device
            )
        
        # Fusion
        fused, gate_values = self.fusion(
            h_acoustic, h_linguistic, return_gate=True
        )  # (B, T, fusion_dim)
        
        # Classification
        logits = self.classifier(fused)  # (B, T, num_classes)
        probs = torch.softmax(logits, dim=-1)
        
        output = {
            'logits': logits,
            'probs': probs,
            'gate_values': gate_values
        }
        
        if return_features:
            output['h_acoustic'] = h_acoustic
            output['h_linguistic'] = h_linguistic
            output['h_fused'] = fused
            
        return output
    
    def predict_frame(
        self,
        audio_features: torch.Tensor,
        h_linguistic: Optional[torch.Tensor] = None
    ) -> Tuple[str, float]:
        """
        Single-frame prediction for real-time inference.
        
        Returns:
            (label, confidence) tuple
        """
        with torch.no_grad():
            h_acoustic = self.acoustic_branch(audio_features)
            
            if h_linguistic is None:
                B = audio_features.size(0)
                h_linguistic = torch.zeros(
                    B, self.fusion.acoustic_transform.in_features,
                    device=audio_features.device
                )
            
            fused, _ = self.fusion(h_acoustic, h_linguistic, return_gate=False)
            logits = self.classifier(fused)
            probs = torch.softmax(logits, dim=-1)
            
            # Get last frame prediction
            last_probs = probs[:, -1, :]  # (B, num_classes)
            pred_idx = last_probs.argmax(dim=-1).item()
            confidence = last_probs[0, pred_idx].item()
            
            return self.LABELS[pred_idx], confidence
```

---

## 🏋️ Phase 4: Training (Week 7-8)

### 4.1 Training Configuration

#### [NEW] [configs/training_config.yaml](file:///f:/Viet-Turn/configs/training_config.yaml)

```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 1e-5
  warmup_steps: 1000
  gradient_clip: 1.0
  
optimizer:
  type: "adamw"
  betas: [0.9, 0.999]
  
scheduler:
  type: "cosine"
  min_lr: 1e-6

loss:
  type: "focal"
  gamma: 2.0  # Focus on hard examples
  alpha: [0.5, 0.3, 0.2]  # Class weights: yield, hold, backchannel

augmentation:
  modality_dropout:
    enabled: true
    text_dropout_prob: 0.3  # Drop text 30% of time
  time_masking:
    enabled: true
    max_mask_length: 10
  noise_injection:
    enabled: true
    snr_range: [10, 30]

early_stopping:
  patience: 10
  metric: "val_f1"
  mode: "max"

checkpointing:
  save_every: 5
  save_best: true
  keep_last: 3
```

### 4.2 Focal Loss Implementation

#### [NEW] [src/training/losses.py](file:///f:/Viet-Turn/src/training/losses.py)

```python
"""
Loss functions for turn-taking prediction.
Focal Loss for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Focuses learning on hard examples, down-weights easy ones.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        self.reduction = reduction
        
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, T, C) or (B, C) - Logits
            targets: (B, T) or (B,) - Class indices
        """
        # Flatten if needed
        if inputs.dim() == 3:
            B, T, C = inputs.shape
            inputs = inputs.reshape(-1, C)
            targets = targets.reshape(-1)
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        probs = F.softmax(inputs, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            focal_weight = focal_weight * alpha_t
        
        # Compute focal loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
```

---

## 🚀 Phase 5: Edge Deployment (Week 9-10)

### 5.1 ONNX Export & Quantization

#### [NEW] [src/inference/onnx_export.py](file:///f:/Viet-Turn/src/inference/onnx_export.py)

```python
"""
Export VietTurnEdge model to ONNX for edge deployment.
"""

import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_acoustic_branch_to_onnx(
    model,
    output_path: str,
    input_shape: tuple = (1, 42, 100),  # (B, features, T)
    opset_version: int = 14
):
    """Export acoustic branch (TCN) to ONNX."""
    model.eval()
    
    dummy_input = torch.randn(*input_shape)
    
    torch.onnx.export(
        model.acoustic_branch,
        dummy_input,
        output_path,
        input_names=['audio_features'],
        output_names=['acoustic_embedding'],
        dynamic_axes={
            'audio_features': {0: 'batch', 2: 'time'},
            'acoustic_embedding': {0: 'batch', 1: 'time'}
        },
        opset_version=opset_version
    )
    
    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Exported acoustic branch to {output_path}")
    
    return output_path

def quantize_onnx_model(
    model_path: str,
    output_path: str,
    quant_type: str = "int8"
):
    """Quantize ONNX model to INT8."""
    qtype = QuantType.QUInt8 if quant_type == "int8" else QuantType.QInt8
    
    quantize_dynamic(
        model_path,
        output_path,
        weight_type=qtype
    )
    
    # Compare sizes
    import os
    original_size = os.path.getsize(model_path) / 1024 / 1024
    quantized_size = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"Original: {original_size:.2f} MB")
    print(f"Quantized: {quantized_size:.2f} MB")
    print(f"Compression: {original_size/quantized_size:.1f}x")
    
    return output_path
```

### 5.2 Real-time Inference Pipeline

#### [NEW] [src/inference/realtime_pipeline.py](file:///f:/Viet-Turn/src/inference/realtime_pipeline.py)

```python
"""
Real-time turn-taking prediction pipeline for edge devices.
"""

import numpy as np
import threading
import queue
from typing import Callable, Optional
import onnxruntime as ort

class RealtimeTurnPredictor:
    """
    Real-time turn prediction with streaming audio.
    
    Target: <100ms latency on Raspberry Pi 4.
    """
    
    def __init__(
        self,
        acoustic_model_path: str,
        fusion_model_path: str,
        vosk_model_path: str,
        sample_rate: int = 16000,
        frame_length_ms: int = 20,
        frame_shift_ms: int = 10,
        decision_threshold: float = 0.7
    ):
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * frame_length_ms / 1000)
        self.frame_shift = int(sample_rate * frame_shift_ms / 1000)
        self.threshold = decision_threshold
        
        # Load ONNX models
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        self.acoustic_session = ort.InferenceSession(
            acoustic_model_path, sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Audio buffer
        self.audio_buffer = np.zeros(self.frame_length * 10, dtype=np.float32)
        
        # ASR results buffer
        self.latest_text = ""
        self.text_embedding = None
        
        # State
        self.running = False
        
    def process_audio_frame(self, audio_chunk: np.ndarray) -> dict:
        """
        Process a single audio frame and return prediction.
        
        Returns:
            {
                'label': 'turn_yield' | 'turn_hold' | 'backchannel',
                'confidence': float,
                'latency_ms': float
            }
        """
        import time
        start_time = time.perf_counter()
        
        # Update buffer
        self.audio_buffer = np.roll(self.audio_buffer, -len(audio_chunk))
        self.audio_buffer[-len(audio_chunk):] = audio_chunk
        
        # Extract features (simplified - would use AudioProcessor)
        # Here: pseudo-code for feature extraction
        features = self._extract_features(self.audio_buffer)
        
        # Run acoustic model
        acoustic_input = features.reshape(1, -1, features.shape[-1])
        acoustic_output = self.acoustic_session.run(
            None, {'audio_features': acoustic_input.astype(np.float32)}
        )[0]
        
        # Get prediction (last frame)
        probs = self._softmax(acoustic_output[0, -1, :])
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        labels = ['turn_yield', 'turn_hold', 'backchannel']
        
        latency = (time.perf_counter() - start_time) * 1000
        
        return {
            'label': labels[pred_idx],
            'confidence': float(confidence),
            'latency_ms': latency
        }
    
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Quick feature extraction for real-time."""
        # Simplified Mel extraction
        # In production: use optimized librosa or custom C++ implementation
        import librosa
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=40,
            hop_length=self.frame_shift, n_fft=512
        )
        return librosa.power_to_db(mel)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
```

---

## ✅ Verification Plan

### Automated Tests

```bash
# 1. Unit Tests - Audio Processing
pytest tests/test_audio_processor.py -v
# Verifies: Mel extraction, F0 extraction, feature alignment

# 2. Unit Tests - Models
pytest tests/test_models.py -v
# Verifies: TCN forward pass, TinyBERT forward, GMU fusion, full model

# 3. Integration Tests - ASR
pytest tests/test_asr.py -v
# Verifies: Vosk streaming, PhoWhisper inference

# 4. End-to-End Pipeline Test
pytest tests/test_pipeline.py -v
# Verifies: Audio → Features → Model → Prediction flow

# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html
```

### Manual Verification

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Run `python scripts/demo.py --mode record` | Record 5s audio, display predictions in real-time |
| 2 | Check latency output | Should show < 100ms per frame on dev machine |
| 3 | Test with sample Vietnamese audio | Labels should match expected turn patterns |
| 4 | Run on Raspberry Pi 4 | Latency < 100ms, no crashes |

### Benchmark Script

```bash
# Run latency benchmark
python scripts/benchmark.py --model deploy/models/viet_turn_edge_int8.onnx --device cpu

# Expected output:
# Average latency: XX.X ms
# P50 latency: XX.X ms
# P90 latency: XX.X ms
# P99 latency: XX.X ms
```

---

## 📅 Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1: Foundation** | Week 1 | Project structure, configs, environment |
| **Phase 2: Data Pipeline** | Week 2-3 | Data download, feature extraction, LLM labeling |
| **Phase 3: Model Development** | Week 4-5 | TCN, PhoBERT, GMU, full model |
| **Phase 4: Training & Eval** | Week 6-7 | Trained model, ablations, results |

**Total: 7 weeks**

---

## ✅ Plan Approved

**Technology Stack Confirmed:**
- ✅ ASR: PhoWhisper-base (sliding window pseudo-streaming)
- ✅ Linguistic: PhoBERT-base-v2 (full model)
- ✅ LLM Labeling: Gemini API

**Ready to proceed with implementation!**
