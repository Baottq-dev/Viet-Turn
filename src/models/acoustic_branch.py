"""
Causal Dilated Temporal Convolutional Network for acoustic features.
Optimized for Vietnamese turn-taking prediction.
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
    
    def get_receptive_field(self) -> int:
        """Calculate receptive field in frames."""
        kernel_size = 3
        dilations = [2 ** i for i in range(len(self.tcn_layers))]
        rf = sum((kernel_size - 1) * d for d in dilations) * 2
        return rf
