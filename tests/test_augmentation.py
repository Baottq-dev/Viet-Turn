"""Tests for data augmentation."""

import torch
import pytest

from src.training.augmentation import VAPAugmenter


class TestVAPAugmenter:
    def _make_batch(self):
        return {
            "audio_waveform": torch.randn(4, 16000 * 2),  # 2s audio
            "texts": ["xin chào", "test hai", "ba bốn", "năm sáu"],
            "vap_labels": torch.randint(0, 256, (4, 100)),
        }

    def test_from_config_none(self):
        aug = VAPAugmenter.from_config(None)
        assert aug.text_dropout_prob == 0.0
        assert not aug.time_mask_enabled
        assert not aug.noise_enabled

    def test_from_config_full(self):
        cfg = {
            "modality_dropout": {"enabled": True, "text_dropout_prob": 0.5},
            "time_masking": {"enabled": True, "max_mask_frames": 25},
            "noise_injection": {"enabled": True, "snr_range": [10, 30]},
        }
        aug = VAPAugmenter.from_config(cfg)
        assert aug.text_dropout_prob == 0.5
        assert aug.time_mask_enabled
        assert aug.noise_enabled

    def test_text_dropout(self):
        aug = VAPAugmenter(text_dropout_prob=1.0)
        texts = ["hello", "world", "test"]
        result = aug.apply_text_dropout(texts)
        assert all(t == "" for t in result)

    def test_text_no_dropout(self):
        aug = VAPAugmenter(text_dropout_prob=0.0)
        texts = ["hello", "world"]
        result = aug.apply_text_dropout(texts)
        assert result == texts

    def test_time_masking_shape(self):
        aug = VAPAugmenter(time_mask_enabled=True, max_mask_frames=10)
        audio = torch.randn(2, 16000)
        result = aug.apply_time_masking(audio)
        assert result.shape == audio.shape

    def test_time_masking_has_zeros(self):
        aug = VAPAugmenter(time_mask_enabled=True, max_mask_frames=25)
        audio = torch.ones(2, 16000)
        result = aug.apply_time_masking(audio)
        # Should have some zeros from masking
        assert (result == 0).any()

    def test_noise_injection_shape(self):
        aug = VAPAugmenter(noise_enabled=True, snr_range=(10, 30))
        audio = torch.randn(2, 16000)
        result = aug.apply_noise_injection(audio)
        assert result.shape == audio.shape

    def test_noise_injection_modifies(self):
        aug = VAPAugmenter(noise_enabled=True, snr_range=(10, 30))
        audio = torch.randn(2, 16000)
        result = aug.apply_noise_injection(audio)
        # Should not be identical
        assert not torch.allclose(audio, result)

    def test_call_full_pipeline(self):
        aug = VAPAugmenter(
            text_dropout_prob=0.5,
            time_mask_enabled=True,
            max_mask_frames=10,
            noise_enabled=True,
            snr_range=(15, 25),
        )
        batch = self._make_batch()
        result = aug(batch, use_text=True)
        assert "audio_waveform" in result
        assert "texts" in result
        assert result["audio_waveform"].shape == (4, 16000 * 2)

    def test_disabled_augmentation_passthrough(self):
        aug = VAPAugmenter()  # all disabled
        batch = self._make_batch()
        original_audio = batch["audio_waveform"].clone()
        result = aug(batch, use_text=True)
        assert torch.allclose(result["audio_waveform"], original_audio)
