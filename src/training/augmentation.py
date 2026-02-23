"""
Data augmentation for MM-VAP-VI training.

Three augmentation strategies from config:
- Modality dropout: randomly drop text input (force audio-only)
- Time masking: mask random contiguous frames in audio (SpecAugment-style)
- Noise injection: add Gaussian noise at random SNR
"""

import torch
import random
from typing import Dict, Optional


class VAPAugmenter:
    """
    Applies augmentation to a training batch.

    Usage:
        augmenter = VAPAugmenter.from_config(config["augmentation"])
        batch = augmenter(batch, use_text=True)
    """

    def __init__(
        self,
        text_dropout_prob: float = 0.0,
        time_mask_enabled: bool = False,
        max_mask_frames: int = 25,
        noise_enabled: bool = False,
        snr_range: tuple = (10, 30),
    ):
        self.text_dropout_prob = text_dropout_prob
        self.time_mask_enabled = time_mask_enabled
        self.max_mask_frames = max_mask_frames
        self.noise_enabled = noise_enabled
        self.snr_min, self.snr_max = snr_range

    @classmethod
    def from_config(cls, aug_config: Optional[Dict]) -> "VAPAugmenter":
        """Build augmenter from config dict."""
        if aug_config is None:
            return cls()

        md = aug_config.get("modality_dropout", {})
        tm = aug_config.get("time_masking", {})
        ni = aug_config.get("noise_injection", {})

        return cls(
            text_dropout_prob=md.get("text_dropout_prob", 0.0) if md.get("enabled", False) else 0.0,
            time_mask_enabled=tm.get("enabled", False),
            max_mask_frames=tm.get("max_mask_frames", 25),
            noise_enabled=ni.get("enabled", False),
            snr_range=tuple(ni.get("snr_range", [10, 30])),
        )

    def apply_text_dropout(self, texts: list) -> list:
        """
        Randomly replace text with empty string.

        Forces the model to sometimes rely on audio-only,
        improving robustness when ASR is unavailable.
        """
        if self.text_dropout_prob <= 0:
            return texts
        return [
            "" if random.random() < self.text_dropout_prob else t
            for t in texts
        ]

    def apply_time_masking(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Mask a random contiguous region in each audio waveform.

        SpecAugment-style: replace a random time region with zeros.
        Operates on raw waveform (B, num_samples).

        Args:
            audio: (B, num_samples) raw waveform.

        Returns:
            Augmented audio with masked regions.
        """
        if not self.time_mask_enabled:
            return audio

        B, T = audio.shape
        # Convert frame-level mask to sample-level (50fps -> 16kHz)
        max_mask_samples = self.max_mask_frames * 320  # 20ms * 16kHz = 320 samples/frame

        audio = audio.clone()
        for i in range(B):
            mask_len = random.randint(1, max_mask_samples)
            if mask_len >= T:
                continue
            start = random.randint(0, T - mask_len)
            audio[i, start:start + mask_len] = 0.0

        return audio

    def apply_noise_injection(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise at a random SNR level.

        Args:
            audio: (B, num_samples) raw waveform.

        Returns:
            Noisy audio.
        """
        if not self.noise_enabled:
            return audio

        audio = audio.clone()
        B = audio.shape[0]

        for i in range(B):
            signal = audio[i]
            signal_power = (signal ** 2).mean()

            if signal_power < 1e-10:
                continue

            snr_db = random.uniform(self.snr_min, self.snr_max)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = torch.randn_like(signal) * noise_power.sqrt()
            audio[i] = signal + noise

        return audio

    def __call__(self, batch: Dict, use_text: bool = True) -> Dict:
        """
        Apply all augmentations to a batch.

        Args:
            batch: Dict with "audio_waveform", "texts", etc.
            use_text: Whether text modality is active (skip text dropout if False).

        Returns:
            Augmented batch (modified in-place keys).
        """
        # Text dropout
        if use_text and self.text_dropout_prob > 0:
            batch["texts"] = self.apply_text_dropout(batch["texts"])

        # Time masking on audio
        if self.time_mask_enabled:
            batch["audio_waveform"] = self.apply_time_masking(batch["audio_waveform"])

        # Noise injection on audio
        if self.noise_enabled:
            batch["audio_waveform"] = self.apply_noise_injection(batch["audio_waveform"])

        return batch
