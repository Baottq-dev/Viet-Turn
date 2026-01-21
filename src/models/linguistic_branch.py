"""
PhoBERT Encoder for Vietnamese linguistic turn-taking features.
Focuses on detecting hư từ (discourse markers) for turn prediction.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, List, Set


class HuTuDetector(nn.Module):
    """
    Lightweight hư từ (discourse marker) detector.
    Provides explicit signal for Vietnamese turn-taking markers.
    """
    
    # Vietnamese discourse markers for turn prediction
    YIELD_MARKERS: Set[str] = {'nhé', 'nhỉ', 'à', 'hả', 'ạ', 'không', 'chứ', 'hen', 'nha'}
    HOLD_MARKERS: Set[str] = {'mà', 'thì', 'là', 'nhưng', 'vì', 'nên', 'nếu', 'khi', 'rồi'}
    BACKCHANNEL_MARKERS: Set[str] = {'ừ', 'vâng', 'ờ', 'dạ', 'ừm', 'ừ hử', 'thế à', 'vậy hả'}
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.all_markers = list(
            self.YIELD_MARKERS | self.HOLD_MARKERS | self.BACKCHANNEL_MARKERS
        )
        self.marker_to_idx = {m: i for i, m in enumerate(self.all_markers)}
        self.num_markers = len(self.all_markers)
        
        # Embedding for each marker type
        self.marker_embed = nn.Embedding(self.num_markers + 1, embedding_dim)
        
        # Classification head for marker type
        self.marker_type_embed = nn.Embedding(4, embedding_dim)  # yield, hold, bc, none
        
    def get_marker_type(self, token: str) -> int:
        """Get marker type: 0=yield, 1=hold, 2=backchannel, 3=none"""
        token = token.lower()
        if token in self.YIELD_MARKERS:
            return 0
        elif token in self.HOLD_MARKERS:
            return 1
        elif token in self.BACKCHANNEL_MARKERS:
            return 2
        return 3
        
    def forward(self, text: str, device: torch.device = None) -> torch.Tensor:
        """Detect markers and return embedding."""
        tokens = text.lower().split()
        
        # Get marker embeddings
        marker_ids = []
        marker_types = []
        
        for token in tokens:
            if token in self.marker_to_idx:
                marker_ids.append(self.marker_to_idx[token])
                marker_types.append(self.get_marker_type(token))
        
        if not marker_ids:
            # No marker found - use default
            marker_ids = [self.num_markers]
            marker_types = [3]
        
        # Create tensors on correct device
        if device is None:
            device = self.marker_embed.weight.device
            
        ids_tensor = torch.LongTensor(marker_ids).to(device)
        types_tensor = torch.LongTensor(marker_types).to(device)
        
        # Combine marker and type embeddings
        marker_embed = self.marker_embed(ids_tensor).mean(dim=0)
        type_embed = self.marker_type_embed(types_tensor).mean(dim=0)
        
        return marker_embed + type_embed


class PhoBERTEncoder(nn.Module):
    """
    Full PhoBERT encoder for Vietnamese turn-taking prediction.
    Uses vinai/phobert-base-v2 with optional marker detection.
    """
    
    def __init__(
        self,
        pretrained: str = "vinai/phobert-base-v2",
        output_dim: int = 64,
        freeze_embeddings: bool = True,
        use_marker_detection: bool = True,
        marker_embedding_dim: int = 32
    ):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.phobert = AutoModel.from_pretrained(pretrained)
        
        # Freeze embeddings for efficiency
        if freeze_embeddings:
            for param in self.phobert.embeddings.parameters():
                param.requires_grad = False
            
        hidden_size = self.phobert.config.hidden_size  # 768
        
        # Marker detector
        self.use_markers = use_marker_detection
        if use_marker_detection:
            self.marker_detector = HuTuDetector(embedding_dim=marker_embedding_dim)
            projection_input = hidden_size + marker_embedding_dim
        else:
            projection_input = hidden_size
            
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(projection_input, projection_input // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_input // 2, output_dim)
        )
        
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
        # Run PhoBERT
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pool: use [CLS] token
        pooled = outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)
        
        # Add marker features
        if self.use_markers and text is not None:
            marker_embed = self.marker_detector(text, device=pooled.device)  # (marker_dim,)
            marker_embed = marker_embed.unsqueeze(0).expand(pooled.size(0), -1)
            pooled = torch.cat([pooled, marker_embed], dim=-1)
        elif self.use_markers:
            # Pad with zeros if no text provided
            B = pooled.size(0)
            marker_dim = self.marker_detector.marker_embed.embedding_dim
            padding = torch.zeros(B, marker_dim, device=pooled.device)
            pooled = torch.cat([pooled, padding], dim=-1)
        
        # Project to output dim
        output = self.output_proj(pooled)  # (B, output_dim)
        
        return output
    
    def encode_text(self, text: str, max_length: int = 128) -> torch.Tensor:
        """Tokenize and encode a text string."""
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        output = self.forward(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            text=text
        )
        
        return output
