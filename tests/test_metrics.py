"""Tests for evaluation metrics."""

import numpy as np
import torch
import pytest

from src.evaluation.frame_metrics import (
    compute_frame_ce,
    compute_perplexity,
    compute_topk_accuracy,
    compute_weighted_f1,
    compute_ece,
    compute_brier_score,
    compute_frame_metrics,
)
from src.evaluation.event_metrics import compute_event_metrics
from src.evaluation.latency_metrics import (
    compute_eot_latency,
    compute_fpr_at_thresholds,
    compute_mst_fpr_curve,
)


class TestFrameMetrics:
    def _make_data(self, B=2, T=50, C=256):
        logits = torch.randn(B, T, C)
        labels = torch.randint(0, C, (B, T))
        # Some invalid frames
        labels[:, -10:] = -1
        return logits, labels

    def test_ce_valid(self):
        logits, labels = self._make_data()
        ce = compute_frame_ce(logits, labels)
        assert isinstance(ce, float)
        assert ce > 0

    def test_ce_all_invalid(self):
        logits = torch.randn(2, 10, 256)
        labels = torch.full((2, 10), -1)
        ce = compute_frame_ce(logits, labels)
        assert np.isnan(ce)

    def test_perplexity(self):
        assert compute_perplexity(1.0) == pytest.approx(np.e, rel=1e-5)
        assert np.isnan(compute_perplexity(float("nan")))

    def test_top1_accuracy_range(self):
        logits, labels = self._make_data()
        acc = compute_topk_accuracy(logits, labels, k=1)
        assert 0 <= acc <= 1

    def test_top5_geq_top1(self):
        logits, labels = self._make_data()
        top1 = compute_topk_accuracy(logits, labels, k=1)
        top5 = compute_topk_accuracy(logits, labels, k=5)
        assert top5 >= top1

    def test_weighted_f1_range(self):
        logits, labels = self._make_data()
        f1 = compute_weighted_f1(logits, labels)
        assert 0 <= f1 <= 1

    def test_ece_range(self):
        logits, labels = self._make_data()
        ece = compute_ece(logits, labels)
        assert 0 <= ece <= 1

    def test_ece_perfect_calibration(self):
        """Perfect predictions should have low ECE."""
        B, T, C = 1, 100, 4
        labels = torch.randint(0, C, (B, T))
        # Create logits that strongly predict the correct class
        logits = torch.full((B, T, C), -10.0)
        for b in range(B):
            for t in range(T):
                logits[b, t, labels[b, t]] = 10.0
        ece = compute_ece(logits, labels)
        assert ece < 0.1

    def test_brier_range(self):
        logits, labels = self._make_data()
        brier = compute_brier_score(logits, labels)
        assert 0 <= brier <= 2

    def test_compute_frame_metrics_keys(self):
        logits, labels = self._make_data()
        metrics = compute_frame_metrics(logits, labels)
        expected_keys = [
            "frame_ce", "frame_perplexity", "frame_top1_acc",
            "frame_top5_acc", "frame_weighted_f1", "frame_ece", "frame_brier",
        ]
        for k in expected_keys:
            assert k in metrics


class TestEventMetrics:
    def test_basic(self):
        gt = np.array([0, 0, 1, 1, 0, 2, 0])
        pred = np.array([0, 0, 1, 0, 0, 2, 0])
        p_shift = np.array([0.1, 0.2, 0.8, 0.6, 0.1, 0.3, 0.1])
        gt_shift = (gt == 1).astype(np.float32)

        results = compute_event_metrics(gt, pred, p_shift, gt_shift)
        assert "shift_hold_ba" in results
        assert "bc_f1" in results
        assert "predict_shift_auc" in results

    def test_all_same_class(self):
        gt = np.zeros(100, dtype=np.int64)
        pred = np.zeros(100, dtype=np.int64)
        results = compute_event_metrics(gt, pred)
        assert np.isnan(results["shift_hold_ba"])


class TestLatencyMetrics:
    def test_eot_latency(self):
        p_shift = np.zeros(200)
        p_shift[50:60] = 0.8  # predict shift around frame 50
        regions = [(45, 55)]
        results = compute_eot_latency(p_shift, regions, threshold=0.5)
        assert results["detection_rate"] == 1.0
        assert results["eot_latency_mean_ms"] >= 0

    def test_eot_no_detection(self):
        p_shift = np.zeros(200)  # never predicts shift
        regions = [(50, 60)]
        results = compute_eot_latency(p_shift, regions, threshold=0.5)
        assert results["detection_rate"] == 0.0

    def test_fpr_at_thresholds(self):
        p_shift = np.random.rand(500)
        gt_shift = np.zeros(500)
        gt_shift[100:120] = 1
        results = compute_fpr_at_thresholds(p_shift, gt_shift)
        for k in results:
            assert k.startswith("fpr_at_")
            assert 0 <= results[k] <= 1

    def test_mst_fpr_curve(self):
        p_shift = np.random.rand(500)
        gt_shift = np.zeros(500)
        gt_shift[100:120] = 1
        results = compute_mst_fpr_curve(p_shift, gt_shift)
        assert "mst_fpr_auc" in results
        assert "mst_values_ms" in results
        assert "fpr_values" in results
        assert len(results["mst_values_ms"]) == len(results["fpr_values"])
