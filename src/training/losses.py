"""
Loss functions for turn-taking prediction.
Includes Focal Loss for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Focuses learning on hard examples, down-weights easy ones.
    Essential for turn-taking where events are rare.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[List[float]] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            gamma: Focusing parameter. Higher = more focus on hard examples
            alpha: Class weights. If None, uniform weights
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha))
        else:
            self.alpha = None
        
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
        
        # Compute cross entropy (without reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        probs = F.softmax(inputs, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply class weights if provided
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


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross entropy loss.
    Helps with overconfident predictions.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) or (B, T, C) - Logits
            targets: (B,) or (B, T) - Class indices
        """
        # Flatten if needed
        if inputs.dim() == 3:
            B, T, C = inputs.shape
            inputs = inputs.reshape(-1, C)
            targets = targets.reshape(-1)
        
        # Create smoothed targets
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)
        
        one_hot = torch.zeros_like(inputs).scatter_(
            1, targets.unsqueeze(1), confidence
        )
        one_hot += smooth_value
        one_hot.scatter_(1, targets.unsqueeze(1), confidence)
        
        # Compute loss
        log_probs = F.log_softmax(inputs, dim=-1)
        loss = -(one_hot * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """Combine multiple losses with weights."""
    
    def __init__(
        self,
        focal_weight: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[List[float]] = None,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.focal_weight = focal_weight
        
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        return self.focal_weight * self.focal(inputs, targets)
