"""
Fusion modules for MM-VAP-VI.

Three fusion strategies for combining acoustic and linguistic modalities:
- CrossAttentionFusion: Bidirectional cross-attention (recommended)
- GMUFusion: Gated Multimodal Unit (lightweight)
- BottleneckFusion: Perceiver-style bottleneck (efficient for long sequences)

All modules share the same interface:
    Input:  acoustic (B, T, dim), linguistic (B, dim)
    Output: fused (B, T, dim)
"""

import torch
import torch.nn as nn
from typing import Optional


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion.

    Audio attends to text (audio queries, text keys/values) and
    text attends to audio (text queries, audio keys/values).
    Results are combined with residual connections.

    Input:
        acoustic: (B, T, dim)  — temporal acoustic features
        linguistic: (B, dim)   — static linguistic features (broadcast to T)

    Output:
        fused: (B, T, dim)
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # Audio → Text attention: audio queries attend to text keys/values
        self.audio_to_text = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Text → Audio attention: text queries attend to audio keys/values
        self.text_to_audio = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms
        self.norm_a2t = nn.LayerNorm(dim)
        self.norm_t2a = nn.LayerNorm(dim)

        # Gating: learn how much to mix cross-attention with original
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        acoustic: torch.Tensor,
        linguistic: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            acoustic: (B, T, dim) temporal acoustic features.
            linguistic: (B, dim) static linguistic features.
            causal_mask: (T, T) optional causal attention mask.

        Returns:
            fused: (B, T, dim) fused representation.
        """
        B, T, D = acoustic.shape

        # Broadcast linguistic to time dimension: (B, dim) -> (B, T, dim)
        ling_expanded = linguistic.unsqueeze(1).expand(B, T, D)

        # Audio-to-text cross attention
        # Audio queries attend to linguistic keys/values
        a2t_out, _ = self.audio_to_text(
            query=acoustic,
            key=ling_expanded,
            value=ling_expanded,
        )
        a2t_out = self.norm_a2t(acoustic + self.dropout(a2t_out))

        # Text-to-audio cross attention
        # Linguistic queries attend to audio keys/values (with causal mask)
        t2a_out, _ = self.text_to_audio(
            query=ling_expanded,
            key=acoustic,
            value=acoustic,
            attn_mask=causal_mask,
        )
        t2a_out = self.norm_t2a(ling_expanded + self.dropout(t2a_out))

        # Gated combination
        gate_input = torch.cat([a2t_out, t2a_out], dim=-1)  # (B, T, 2*dim)
        gate_weight = self.gate(gate_input)  # (B, T, dim) values in [0, 1]

        fused = gate_weight * a2t_out + (1 - gate_weight) * t2a_out

        return fused


class GMUFusion(nn.Module):
    """
    Gated Multimodal Unit (GMU) fusion.

    Lightweight fusion via a learned gate:
        z = sigmoid(W_z · [h_a; h_l] + b_z)
        h_fused = z * tanh(W_a · h_a) + (1-z) * tanh(W_l · h_l)

    ~260K params for dim=256.

    Input:
        acoustic: (B, T, dim)
        linguistic: (B, dim)

    Output:
        fused: (B, T, dim)
    """

    def __init__(
        self,
        dim: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim

        # Transform each modality
        self.W_a = nn.Linear(dim, dim)
        self.W_l = nn.Linear(dim, dim)

        # Gate: decides mixing ratio
        self.W_z = nn.Linear(dim * 2, dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        acoustic: torch.Tensor,
        linguistic: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            acoustic: (B, T, dim)
            linguistic: (B, dim)
            causal_mask: Unused, kept for interface compatibility.

        Returns:
            fused: (B, T, dim)
        """
        B, T, D = acoustic.shape

        # Broadcast linguistic: (B, dim) -> (B, T, dim)
        ling_expanded = linguistic.unsqueeze(1).expand(B, T, D)

        # Transform
        h_a = torch.tanh(self.W_a(acoustic))
        h_l = torch.tanh(self.W_l(ling_expanded))

        # Gate
        z = torch.sigmoid(self.W_z(torch.cat([acoustic, ling_expanded], dim=-1)))

        # Fuse
        fused = z * h_a + (1 - z) * h_l
        fused = self.norm(fused + acoustic)  # residual from acoustic
        fused = self.dropout(fused)

        return fused


class BottleneckFusion(nn.Module):
    """
    Perceiver-style bottleneck fusion.

    Uses a small set of learnable latent tokens to mediate between modalities.
    More efficient than full cross-attention for long sequences.

    Pipeline:
        1. Latent tokens cross-attend to [acoustic; linguistic] (compress)
        2. Acoustic cross-attends to latent tokens (decompress)

    ~800K params for dim=256, num_latents=16.

    Input:
        acoustic: (B, T, dim)
        linguistic: (B, dim)

    Output:
        fused: (B, T, dim)
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        num_latents: int = 16,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents

        # Learnable latent tokens
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)

        # Compress: latents attend to input
        self.compress_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.compress_norm = nn.LayerNorm(dim)
        self.compress_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.compress_ffn_norm = nn.LayerNorm(dim)

        # Decompress: acoustic attends to latents
        self.decompress_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.decompress_norm = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        acoustic: torch.Tensor,
        linguistic: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            acoustic: (B, T, dim)
            linguistic: (B, dim)
            causal_mask: Unused, kept for interface compatibility.

        Returns:
            fused: (B, T, dim)
        """
        B, T, D = acoustic.shape

        # Broadcast linguistic: (B, dim) -> (B, 1, dim)
        ling_expanded = linguistic.unsqueeze(1)

        # Concat input: [acoustic; linguistic] -> (B, T+1, dim)
        combined = torch.cat([acoustic, ling_expanded], dim=1)

        # Expand latents for batch: (num_latents, dim) -> (B, num_latents, dim)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        # Compress: latents cross-attend to combined input
        compressed, _ = self.compress_attn(
            query=latents,
            key=combined,
            value=combined,
        )
        compressed = self.compress_norm(latents + self.dropout(compressed))
        compressed = self.compress_ffn_norm(compressed + self.compress_ffn(compressed))

        # Decompress: acoustic cross-attends to compressed latents
        decompressed, _ = self.decompress_attn(
            query=acoustic,
            key=compressed,
            value=compressed,
        )
        fused = self.decompress_norm(acoustic + self.dropout(decompressed))

        return fused


def build_fusion(fusion_type: str, **kwargs) -> nn.Module:
    """
    Factory function to build fusion module by type.

    Args:
        fusion_type: "cross_attention", "gmu", or "bottleneck"
        **kwargs: Arguments passed to the fusion constructor.

    Returns:
        Fusion module instance.
    """
    fusion_map = {
        "cross_attention": CrossAttentionFusion,
        "gmu": GMUFusion,
        "bottleneck": BottleneckFusion,
    }
    if fusion_type not in fusion_map:
        raise ValueError(f"Unknown fusion type: {fusion_type}. Choose from {list(fusion_map.keys())}")
    return fusion_map[fusion_type](**kwargs)
