"""
Causal Transformer with ALiBi for MM-VAP-VI.

Processes the fused multimodal features with causal attention
(each frame can only attend to past and current frames).
Uses ALiBi (Attention with Linear Biases) for positional encoding.
Supports KV cache for efficient streaming inference.
"""

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    Compute ALiBi slopes for each attention head.

    Following Press et al. (2022): slopes are geometric sequence
    from 2^(-8/n) to 2^(-8) where n is number of heads.
    """
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = torch.arange(1, closest_power_of_2 + 1)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        extra_powers = torch.arange(1, 2 * (num_heads - closest_power_of_2) + 1, 2)
        extra_slopes = torch.pow(extra_base, extra_powers)
        slopes = torch.cat([slopes, extra_slopes])

    return slopes


def build_alibi_bias(num_heads: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build ALiBi attention bias matrix.

    Returns:
        bias: (num_heads, seq_len, seq_len) to be added to attention scores.
    """
    slopes = get_alibi_slopes(num_heads).to(device)  # (num_heads,)

    # Distance matrix: positions[i] - positions[j]
    positions = torch.arange(seq_len, device=device)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq_len, seq_len)

    # ALiBi bias: slope * distance (negative for causal)
    bias = slopes.unsqueeze(-1).unsqueeze(-1) * distance.unsqueeze(0)  # (H, T, T)

    return bias


def build_alibi_bias_incremental(
    num_heads: int,
    total_len: int,
    new_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build ALiBi bias for incremental (cached) inference.

    Query positions are [total_len - new_len, ..., total_len - 1],
    Key positions are [0, ..., total_len - 1].

    Returns:
        bias: (num_heads, new_len, total_len)
    """
    slopes = get_alibi_slopes(num_heads).to(device)

    query_pos = torch.arange(total_len - new_len, total_len, device=device)
    key_pos = torch.arange(total_len, device=device)
    distance = query_pos.unsqueeze(1) - key_pos.unsqueeze(0)  # (new_len, total_len)

    bias = slopes.unsqueeze(-1).unsqueeze(-1) * distance.unsqueeze(0)  # (H, new_len, total_len)
    return bias


# Type alias for KV cache: list of (key, value) tuples per layer
KVCache = List[Optional[Tuple[torch.Tensor, torch.Tensor]]]


class CausalTransformerLayer(nn.Module):
    """Single causal transformer layer with ALiBi and optional KV cache."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (B, T, dim) — full sequence or new tokens only (if using cache).
            attn_mask: Attention mask (causal + ALiBi).
            kv_cache: Optional (cached_key, cached_value), each (B, T_past, dim).

        Returns:
            output: (B, T, dim)
            new_kv_cache: (key, value) for this layer, or None if not caching.
        """
        # Self-attention with pre-norm
        residual = x
        x_normed = self.norm1(x)

        if kv_cache is not None:
            # Incremental decoding: x is new tokens, prepend cached K/V
            cached_k, cached_v = kv_cache
            # We need to compute K, V for new tokens and concatenate with cache.
            # nn.MultiheadAttention doesn't directly support this, so we
            # construct full key/value sequences manually.
            # Query = new tokens normed, Key/Value = cached + new tokens normed
            # But we need normed past tokens too — so we store post-norm K/V.
            # Alternative: just pass full sequence through and slice.
            # For simplicity with nn.MHA, concatenate cached with new for KV.
            kv_input = torch.cat([cached_k, x_normed], dim=1)
            x_out, _ = self.self_attn(
                query=x_normed,
                key=kv_input,
                value=kv_input,
                attn_mask=attn_mask,
            )
            # Update cache: store all normed inputs for K/V
            new_cache = (kv_input, kv_input)
        else:
            x_out, _ = self.self_attn(x_normed, x_normed, x_normed, attn_mask=attn_mask)
            new_cache = (x_normed, x_normed)

        x = residual + self.dropout(x_out)

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x, new_cache


class CausalTransformer(nn.Module):
    """
    Stack of causal transformer layers with ALiBi positional encoding.
    Supports KV cache for efficient streaming inference.

    Input: (B, T, dim) — fused multimodal features
    Output: (B, T, dim) — contextualized features
    """

    def __init__(
        self,
        num_layers: int = 4,
        dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        use_alibi: bool = True,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_alibi = use_alibi
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.layers = nn.ModuleList([
            CausalTransformerLayer(dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(dim)

    def _build_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build causal mask with optional ALiBi bias."""
        # Causal mask: -inf for future positions
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

        if self.use_alibi:
            alibi = build_alibi_bias(self.num_heads, seq_len, device)
            # Expand causal mask to (num_heads, T, T) and add ALiBi
            causal_mask = causal_mask.unsqueeze(0) + alibi
            return causal_mask  # (H, T, T)

        return causal_mask  # (T, T)

    def _build_incremental_mask(
        self, total_len: int, new_len: int, device: torch.device,
    ) -> torch.Tensor:
        """Build attention mask for incremental (cached) decoding."""
        # New queries can attend to all past + current positions (causal is satisfied by design)
        # Shape: (new_len, total_len) — no future masking needed since new tokens are at the end
        causal_mask = torch.zeros(new_len, total_len, device=device)
        # But we still need causal within the new tokens themselves
        if new_len > 1:
            new_causal = torch.triu(
                torch.full((new_len, new_len), float("-inf"), device=device),
                diagonal=1,
            )
            causal_mask[:, -new_len:] = new_causal

        if self.use_alibi:
            alibi = build_alibi_bias_incremental(self.num_heads, total_len, new_len, device)
            causal_mask = causal_mask.unsqueeze(0) + alibi  # (H, new_len, total_len)

        return causal_mask

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Args:
            x: (B, T, dim) — full sequence or new frames only (when using cache).
            kv_cache: Optional list of per-layer (key, value) cache.
                      Pass None for full-sequence mode (training).

        Returns:
            output: (B, T, dim) — same time dimension as input x.
            new_kv_cache: Updated cache (list of per-layer KV), or None if not caching.
        """
        B, T, D = x.shape
        device = x.device
        use_cache = kv_cache is not None

        if use_cache:
            # Incremental mode
            past_len = kv_cache[0][0].shape[1] if kv_cache[0] is not None else 0
            total_len = past_len + T
            mask = self._build_incremental_mask(total_len, T, device)
            if self.use_alibi:
                mask = mask.unsqueeze(0).expand(B, -1, -1, -1)
                mask = mask.reshape(B * self.num_heads, T, total_len)

            new_kv_cache = []
            for i, layer in enumerate(self.layers):
                layer_cache = kv_cache[i] if kv_cache[i] is not None else None
                x, new_layer_cache = layer(x, attn_mask=mask, kv_cache=layer_cache)
                new_kv_cache.append(new_layer_cache)

            x = self.final_norm(x)
            return x, new_kv_cache
        else:
            # Full-sequence mode (training)
            mask = self._build_mask(T, device)
            if self.use_alibi:
                mask = mask.unsqueeze(0).expand(B, -1, -1, -1)
                mask = mask.reshape(B * self.num_heads, T, T)

            for layer in self.layers:
                if self.use_gradient_checkpointing and self.training:
                    # Checkpoint doesn't support extra returns, use wrapper
                    def layer_fn(layer_mod, x_in, mask_in):
                        out, _ = layer_mod(x_in, attn_mask=mask_in)
                        return out
                    x = checkpoint(layer_fn, layer, x, mask, use_reentrant=False)
                else:
                    x, _ = layer(x, attn_mask=mask)

            x = self.final_norm(x)
            return x, None

    def init_cache(self, num_layers: int = None) -> KVCache:
        """Initialize empty KV cache."""
        n = num_layers or len(self.layers)
        return [None] * n
