"""
VAP Projection Head for MM-VAP-VI.

Maps transformer output to 256-class VAP predictions at each frame.
"""

import torch
import torch.nn as nn


class VAPProjectionHead(nn.Module):
    """
    Projects transformer features to 256-class VAP logits.

    Input: (B, T, dim)
    Output: (B, T, 256)
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)

        Returns:
            logits: (B, T, num_classes)
        """
        return self.head(x)
