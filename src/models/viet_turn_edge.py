"""
VietTurnEdge: Full hybrid model for Vietnamese turn-taking prediction.
Combines TCN acoustic encoder, PhoBERT linguistic encoder, and GMU fusion.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .acoustic_branch import CausalDilatedTCN
from .linguistic_branch import PhoBERTEncoder
from .fusion import GatedMultimodalUnit


class VietTurnEdge(nn.Module):
    """
    Hybrid multimodal model for Vietnamese turn-taking prediction.
    
    Architecture:
        Audio → TCN → GMU → Classifier → Turn Prediction
        Text → PhoBERT ↗
    
    Outputs 3 classes:
        - turn_yield: Speaker finished, ready for listener to speak
        - turn_hold: Speaker will continue
        - backchannel: Short response without taking turn
    """
    
    LABELS = ['turn_yield', 'turn_hold', 'backchannel']
    
    def __init__(
        self,
        acoustic_config: Optional[dict] = None,
        linguistic_config: Optional[dict] = None,
        fusion_config: Optional[dict] = None,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Default configs
        acoustic_config = acoustic_config or {
            'input_dim': 42,
            'hidden_dim': 128,
            'output_dim': 64,
            'num_layers': 4,
            'kernel_size': 3,
            'dropout': 0.1
        }
        
        linguistic_config = linguistic_config or {
            'pretrained': 'vinai/phobert-base-v2',
            'output_dim': 64,
            'freeze_embeddings': True,
            'use_marker_detection': True
        }
        
        fusion_config = fusion_config or {
            'acoustic_dim': 64,
            'linguistic_dim': 64,
            'hidden_dim': 128,
            'output_dim': 64
        }
        
        # Build branches
        self.acoustic_branch = CausalDilatedTCN(**acoustic_config)
        self.linguistic_branch = PhoBERTEncoder(**linguistic_config)
        self.fusion = GatedMultimodalUnit(**fusion_config)
        
        # Classifier head
        fusion_output_dim = fusion_config.get('output_dim', 64)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        self.num_classes = num_classes
        
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
                - logits: (B, T, num_classes)
                - probs: (B, T, num_classes)
                - gate_values: (B, T) - GMU gate values
        """
        B = audio_features.size(0)
        
        # Acoustic branch
        h_acoustic = self.acoustic_branch(audio_features)  # (B, T, acoustic_dim)
        
        # Linguistic branch
        if input_ids is not None:
            h_linguistic = self.linguistic_branch(
                input_ids, attention_mask, text
            )  # (B, linguistic_dim)
        else:
            # No text available - use zero vector
            h_linguistic = torch.zeros(
                B, self.fusion.linguistic_dim,
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
    
    def predict(
        self,
        audio_features: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions.
        
        Returns:
            (predictions, confidences) - Both shape (B, T)
        """
        with torch.no_grad():
            output = self.forward(
                audio_features, input_ids, attention_mask, text
            )
            probs = output['probs']
            
            confidences, predictions = probs.max(dim=-1)
            
            return predictions, confidences
    
    def predict_last_frame(
        self,
        audio_features: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Get prediction for last frame only (for streaming).
        
        Returns:
            (label_name, confidence)
        """
        predictions, confidences = self.predict(
            audio_features, input_ids, attention_mask, text
        )
        
        pred_idx = predictions[0, -1].item()
        confidence = confidences[0, -1].item()
        
        return self.LABELS[pred_idx], confidence
    
    @classmethod
    def from_config(cls, config: dict) -> 'VietTurnEdge':
        """Create model from config dict."""
        return cls(
            acoustic_config=config.get('acoustic_branch', {}),
            linguistic_config=config.get('linguistic_branch', {}),
            fusion_config=config.get('fusion', {}),
            num_classes=config.get('model', {}).get('num_classes', 3)
        )
