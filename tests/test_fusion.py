"""Tests for fusion modules."""

import torch
import pytest

from src.models.fusion import (
    CrossAttentionFusion,
    GMUFusion,
    BottleneckFusion,
    build_fusion,
)


DIM = 64
B = 2
T = 50


@pytest.fixture
def acoustic():
    return torch.randn(B, T, DIM)


@pytest.fixture
def linguistic():
    return torch.randn(B, DIM)


class TestCrossAttentionFusion:
    def test_output_shape(self, acoustic, linguistic):
        fusion = CrossAttentionFusion(dim=DIM, num_heads=4)
        out = fusion(acoustic, linguistic)
        assert out.shape == (B, T, DIM)

    def test_gradient_flow(self, acoustic, linguistic):
        fusion = CrossAttentionFusion(dim=DIM, num_heads=4)
        out = fusion(acoustic, linguistic)
        loss = out.sum()
        loss.backward()
        assert acoustic.grad is None  # not leaf by default
        for p in fusion.parameters():
            assert p.grad is not None


class TestGMUFusion:
    def test_output_shape(self, acoustic, linguistic):
        fusion = GMUFusion(dim=DIM)
        out = fusion(acoustic, linguistic)
        assert out.shape == (B, T, DIM)

    def test_param_count(self):
        fusion = GMUFusion(dim=256)
        n_params = sum(p.numel() for p in fusion.parameters())
        # Should be lightweight (~260K range)
        assert n_params < 500_000


class TestBottleneckFusion:
    def test_output_shape(self, acoustic, linguistic):
        fusion = BottleneckFusion(dim=DIM, num_heads=4, num_latents=8)
        out = fusion(acoustic, linguistic)
        assert out.shape == (B, T, DIM)

    def test_different_num_latents(self, acoustic, linguistic):
        for n in [4, 8, 16, 32]:
            fusion = BottleneckFusion(dim=DIM, num_heads=4, num_latents=n)
            out = fusion(acoustic, linguistic)
            assert out.shape == (B, T, DIM)


class TestBuildFusion:
    def test_cross_attention(self):
        f = build_fusion("cross_attention", dim=DIM, num_heads=4)
        assert isinstance(f, CrossAttentionFusion)

    def test_gmu(self):
        f = build_fusion("gmu", dim=DIM)
        assert isinstance(f, GMUFusion)

    def test_bottleneck(self):
        f = build_fusion("bottleneck", dim=DIM, num_heads=4)
        assert isinstance(f, BottleneckFusion)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown fusion type"):
            build_fusion("nonexistent", dim=DIM)

    def test_all_same_interface(self, acoustic, linguistic):
        """All fusion types should accept same inputs and produce same output shape."""
        for fusion_type in ["cross_attention", "gmu", "bottleneck"]:
            f = build_fusion(fusion_type, dim=DIM, num_heads=4)
            out = f(acoustic, linguistic)
            assert out.shape == (B, T, DIM), f"Failed for {fusion_type}"
