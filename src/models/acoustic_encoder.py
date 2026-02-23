"""
Acoustic Encoder for MM-VAP-VI.

Wraps Wav2Vec2 or WavLM pretrained models to extract acoustic representations
at 50fps from raw audio waveforms.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, WavLMModel


class AcousticEncoder(nn.Module):
    """
    Wav2Vec2/WavLM based acoustic encoder.

    Input: raw waveform (B, num_samples) at 16kHz
    Output: (B, T, output_dim) at 50fps
    """

    def __init__(
        self,
        pretrained: str = "nguyenvulebinh/wav2vec2-base-vi",
        encoder_type: str = "wav2vec2",
        output_dim: int = 256,
        freeze_layers: int = 8,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.encoder_type = encoder_type

        if encoder_type == "wavlm":
            self.encoder = WavLMModel.from_pretrained(pretrained)
        else:
            self.encoder = Wav2Vec2Model.from_pretrained(pretrained)

        self.hidden_size = self.encoder.config.hidden_size  # 768

        # Freeze feature extractor (CNN) always
        self.encoder.feature_extractor._freeze_parameters()

        # Freeze first N transformer layers
        if freeze_layers > 0:
            for i, layer in enumerate(self.encoder.encoder.layers[:freeze_layers]):
                for param in layer.parameters():
                    param.requires_grad = False

        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        # Projection to output dim
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            waveform: (B, num_samples) raw audio at 16kHz.
            attention_mask: (B, num_samples) optional mask for padded audio.

        Returns:
            features: (B, T, output_dim) at ~50fps.
        """
        outputs = self.encoder(
            input_values=waveform,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        hidden = outputs.last_hidden_state  # (B, T, 768)
        features = self.proj(hidden)  # (B, T, output_dim)

        return features

    def freeze_all(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n: int):
        """Unfreeze last N transformer layers + projection."""
        # Unfreeze projection always
        for param in self.proj.parameters():
            param.requires_grad = True

        # Unfreeze last N encoder layers
        total_layers = len(self.encoder.encoder.layers)
        for layer in self.encoder.encoder.layers[total_layers - n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters (except CNN feature extractor)."""
        for param in self.parameters():
            param.requires_grad = True
        # Keep CNN frozen
        self.encoder.feature_extractor._freeze_parameters()
