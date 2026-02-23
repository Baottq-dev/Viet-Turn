"""Tests for VAP label encoding/decoding."""

import numpy as np
import torch
import pytest

from src.utils.labels import (
    encode_vap_labels,
    encode_vap_labels_torch,
    decode_vap_labels,
    decode_vap_labels_torch,
    vap_probs_to_p_now,
    NUM_CLASSES,
    NUM_BINS,
    NUM_SPEAKERS,
    BIN_FRAMES,
    LOOKAHEAD_FRAMES,
)


class TestEncodeDecodeRoundtrip:
    """Verify encode -> decode -> encode is consistent."""

    def test_roundtrip_numpy(self):
        """encode -> decode should produce consistent bit patterns."""
        va_matrix = np.zeros((2, 200), dtype=np.float32)
        va_matrix[0, 10:50] = 1  # Speaker 0 active frames 10-50
        va_matrix[1, 60:90] = 1  # Speaker 1 active frames 60-90

        labels = encode_vap_labels(va_matrix)

        # Valid frames should be 0..99 (200 - 100 lookahead)
        valid_mask = labels >= 0
        assert valid_mask[:100].all()
        assert not valid_mask[100:].any()

        # Decode valid labels
        valid_labels = labels[valid_mask]
        decoded = decode_vap_labels(valid_labels)
        assert decoded.shape == (100, 2, 4)

        # Re-encode from decoded: each bit should match
        for i, label_val in enumerate(valid_labels):
            for s in range(NUM_SPEAKERS):
                for b in range(NUM_BINS):
                    bit_pos = s * NUM_BINS + b
                    expected_bit = (label_val >> bit_pos) & 1
                    assert decoded[i, s, b] == expected_bit

    def test_roundtrip_torch(self):
        """Torch version should produce same results as numpy."""
        rng = np.random.RandomState(42)
        va_np = (rng.rand(2, 300) > 0.5).astype(np.float32)

        labels_np = encode_vap_labels(va_np)
        labels_torch = encode_vap_labels_torch(torch.from_numpy(va_np)).numpy()

        np.testing.assert_array_equal(labels_np, labels_torch)

    def test_decode_torch_matches_numpy(self):
        """Torch decode should match numpy decode."""
        labels = np.array([0, 1, 127, 128, 255], dtype=np.int64)
        decoded_np = decode_vap_labels(labels)
        decoded_torch = decode_vap_labels_torch(torch.from_numpy(labels)).numpy()
        np.testing.assert_array_equal(decoded_np, decoded_torch)


class TestEncodingEdgeCases:
    """Test edge cases in label encoding."""

    def test_all_silence(self):
        """All silence should produce class 0."""
        va_matrix = np.zeros((2, 200), dtype=np.float32)
        labels = encode_vap_labels(va_matrix)
        valid = labels[labels >= 0]
        assert (valid == 0).all()

    def test_all_speaker0(self):
        """All speaker 0 active should set bits 0-3."""
        va_matrix = np.zeros((2, 200), dtype=np.float32)
        va_matrix[0, :] = 1
        labels = encode_vap_labels(va_matrix)
        valid = labels[labels >= 0]
        # Bits 0-3 should be 1 = 0b00001111 = 15
        assert (valid == 15).all()

    def test_all_both_speakers(self):
        """Both speakers always active should give class 255."""
        va_matrix = np.ones((2, 200), dtype=np.float32)
        labels = encode_vap_labels(va_matrix)
        valid = labels[labels >= 0]
        assert (valid == 255).all()

    def test_short_audio(self):
        """Audio shorter than lookahead should give all invalid."""
        va_matrix = np.ones((2, 50), dtype=np.float32)  # 50 < 100 lookahead
        labels = encode_vap_labels(va_matrix)
        assert (labels == -1).all()

    def test_class_range(self):
        """All labels should be in [0, 255] or -1."""
        rng = np.random.RandomState(123)
        va_matrix = (rng.rand(2, 500) > 0.5).astype(np.float32)
        labels = encode_vap_labels(va_matrix)
        valid = labels[labels >= 0]
        assert valid.min() >= 0
        assert valid.max() <= 255


class TestDecoding:
    """Test label decoding."""

    def test_decode_class_0(self):
        """Class 0 = all silent."""
        decoded = decode_vap_labels(np.array([0]))
        assert decoded.shape == (1, 2, 4)
        assert (decoded == 0).all()

    def test_decode_class_255(self):
        """Class 255 = all bits set."""
        decoded = decode_vap_labels(np.array([255]))
        assert (decoded == 1).all()

    def test_decode_class_1(self):
        """Class 1 = only bit 0 set (speaker 0, bin 0)."""
        decoded = decode_vap_labels(np.array([1]))
        assert decoded[0, 0, 0] == 1
        assert decoded[0, 0, 1] == 0
        assert decoded[0, 1, 0] == 0


class TestPNow:
    """Test p_now computation."""

    def test_p_now_shape(self):
        """p_now should have shape (B, T)."""
        B, T = 2, 50
        probs = torch.softmax(torch.randn(B, T, 256), dim=-1)
        p0, p1 = vap_probs_to_p_now(probs)
        assert p0.shape == (B, T)
        assert p1.shape == (B, T)

    def test_p_now_range(self):
        """p_now should be in [0, 1]."""
        probs = torch.softmax(torch.randn(3, 100, 256), dim=-1)
        p0, p1 = vap_probs_to_p_now(probs)
        assert (p0 >= 0).all() and (p0 <= 1).all()
        assert (p1 >= 0).all() and (p1 <= 1).all()

    def test_p_now_uniform(self):
        """Uniform probs should give ~0.5 for each speaker."""
        probs = torch.ones(1, 10, 256) / 256
        p0, p1 = vap_probs_to_p_now(probs)
        # Half of classes have bit 0 set, half don't
        assert torch.allclose(p0, torch.tensor(0.5), atol=0.01)
        assert torch.allclose(p1, torch.tensor(0.5), atol=0.01)
