"""
Loss functions for VAP training.

Frame-level cross-entropy over 256 classes with optional transition weighting
and focal loss for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VAPLoss(nn.Module):
    """
    Frame-level loss for VAP training.

    Computes cross-entropy at every valid frame (label != -1).
    Optionally applies transition weighting to emphasize frames near speaker changes.
    """

    def __init__(
        self,
        num_classes: int = 256,
        transition_weight: float = 1.0,
        transition_window_frames: int = 25,  # ±500ms at 50fps
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        ignore_index: int = -1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.transition_weight = transition_weight
        self.transition_window_frames = transition_window_frames
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction="none",
            label_smoothing=label_smoothing,
        )

    def _detect_transitions(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Detect frames near label transitions (speaker changes).

        Args:
            labels: (B, T) class indices.

        Returns:
            weights: (B, T) per-frame weights. 1.0 for normal frames,
                     transition_weight for frames near transitions.
        """
        B, T = labels.shape
        weights = torch.ones(B, T, device=labels.device)

        if self.transition_weight <= 1.0:
            return weights

        # Detect transitions: frames where label changes between two valid frames
        valid = labels != self.ignore_index
        shifted = torch.roll(labels, 1, dims=1)
        shifted[:, 0] = labels[:, 0]  # No transition at frame 0
        valid_shifted = torch.roll(valid, 1, dims=1)
        valid_shifted[:, 0] = True  # Frame 0 has no previous frame to check

        transitions = (labels != shifted) & valid & valid_shifted  # (B, T) bool

        # Expand transitions by window using max-pooling approach
        # This is equivalent to: for each transition, mark all frames within ±window
        if transitions.any():
            trans_float = transitions.float().unsqueeze(1)  # (B, 1, T)
            kernel_size = 2 * self.transition_window_frames + 1
            padding = self.transition_window_frames
            expanded = torch.nn.functional.max_pool1d(
                trans_float, kernel_size=kernel_size, stride=1, padding=padding
            ).squeeze(1)  # (B, T)
            weights = torch.where(expanded.bool(), self.transition_weight, weights)

        return weights

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        va_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, T, num_classes) model predictions.
            labels: (B, T) target class indices, -1 for invalid.
            va_matrix: (B, 2, T) optional, for transition detection.

        Returns:
            loss: scalar tensor.
        """
        B, T, C = logits.shape

        # Flatten for CE computation
        logits_flat = logits.reshape(-1, C)
        labels_flat = labels.reshape(-1)

        # Base per-frame loss
        per_frame_loss = self.ce(logits_flat, labels_flat)  # (B*T,)

        if self.use_focal:
            # Apply focal weighting
            probs = F.softmax(logits_flat, dim=-1)
            valid_mask = labels_flat != self.ignore_index
            p_t = torch.zeros_like(per_frame_loss)
            if valid_mask.any():
                p_t[valid_mask] = probs[valid_mask].gather(
                    1, labels_flat[valid_mask].unsqueeze(1)
                ).squeeze(1)
            focal_weight = (1 - p_t) ** self.focal_gamma
            per_frame_loss = focal_weight * per_frame_loss

        # Reshape back
        per_frame_loss = per_frame_loss.reshape(B, T)

        # Apply transition weighting
        if self.transition_weight > 1.0:
            weights = self._detect_transitions(labels)
            per_frame_loss = per_frame_loss * weights

        # Average over valid frames only
        valid_mask = (labels != self.ignore_index)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = (per_frame_loss * valid_mask.float()).sum() / valid_mask.float().sum()
        return loss
