"""
Gated Multimodal Unit (GMU) for fusing acoustic and linguistic features.
Learns to dynamically weight each modality based on context.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class GatedMultimodalUnit(nn.Module):
    """
    GMU Fusion Layer.
    
    h = z ⊙ tanh(W_a · h_a) + (1-z) ⊙ tanh(W_t · h_t)
    z = σ(W_z · [h_a; h_t])
    
    Where:
        h_a: Acoustic features
        h_t: Linguistic/Text features  
        z: Learned gate (0=text, 1=audio)
    """
    
    def __init__(
        self,
        acoustic_dim: int = 64,
        linguistic_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64
    ):
        super().__init__()
        
        self.acoustic_dim = acoustic_dim
        self.linguistic_dim = linguistic_dim
        
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fuse acoustic and linguistic representations.
        
        Args:
            h_acoustic: (B, T, acoustic_dim) or (B, acoustic_dim)
            h_linguistic: (B, linguistic_dim) - Usually pooled from BERT
            return_gate: If True, also return gate values for visualization
            
        Returns:
            Fused representation (B, T, output_dim) or (B, output_dim)
            Optional gate values for analysis
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
            gate_value = z.mean(dim=-1)  # Average gate across hidden dim
            return output, gate_value
        return output, None


class AttentionFusion(nn.Module):
    """
    Alternative fusion using cross-attention.
    Query from acoustic, Key/Value from linguistic.
    """
    
    def __init__(
        self,
        acoustic_dim: int = 64,
        linguistic_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        output_dim: int = 64
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.acoustic_proj = nn.Linear(acoustic_dim, hidden_dim)
        self.linguistic_proj = nn.Linear(linguistic_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        h_acoustic: torch.Tensor,
        h_linguistic: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_acoustic: (B, T, acoustic_dim)
            h_linguistic: (B, linguistic_dim)
        """
        B, T, _ = h_acoustic.shape
        
        # Project
        q = self.acoustic_proj(h_acoustic)  # (B, T, hidden)
        
        # Expand linguistic for K, V
        h_ling_expanded = h_linguistic.unsqueeze(1)  # (B, 1, ling_dim)
        k = v = self.linguistic_proj(h_ling_expanded)  # (B, 1, hidden)
        
        # Cross attention
        attn_out, _ = self.attention(q, k, v)  # (B, T, hidden)
        
        # Combine with original acoustic
        combined = attn_out + q
        
        return self.output_proj(combined)
