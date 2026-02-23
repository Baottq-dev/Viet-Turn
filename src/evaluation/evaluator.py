"""
MM-VAP-VI Evaluator - Orchestrates all evaluation tiers.

Runs the full model on a test set and computes:
- Tier 1: Frame-level metrics (CE, perplexity, accuracy)
- Tier 2: Event-level metrics (Shift/Hold BA, BC F1, AUC)
- Tier 3: Latency metrics (EOT latency, FPR, MST-FPR)
- Tier 4: VAQI (Voice Agent Quality Index)
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.model import MMVAPModel
from .frame_metrics import compute_frame_metrics
from .event_metrics import classify_events, compute_event_metrics
from .latency_metrics import compute_eot_latency, compute_fpr_at_thresholds, compute_mst_fpr_curve, compute_vaqi
from .vietnamese_analysis import analyze_marker_impact


class MMVAPEvaluator:
    """
    Evaluator for MM-VAP-VI model.

    Usage:
        evaluator = MMVAPEvaluator(model, test_loader, device="cuda")
        results = evaluator.evaluate()
        evaluator.save_report(results, "outputs/eval_report.json")
    """

    def __init__(
        self,
        model: MMVAPModel,
        test_loader: DataLoader,
        device: str = "cuda",
        shift_threshold: float = 0.5,
        bc_threshold: float = 0.3,
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.shift_threshold = shift_threshold
        self.bc_threshold = bc_threshold

    @torch.no_grad()
    def evaluate(self, use_text: bool = True) -> Dict:
        """
        Run full evaluation.

        Args:
            use_text: Whether to use text modality.

        Returns:
            Dict with all metrics across all tiers.
        """
        # Collect predictions across all batches
        all_logits = []
        all_labels = []
        all_va_matrices = []
        all_texts = []

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            audio = batch["audio_waveform"].to(self.device)
            labels = batch["vap_labels"]
            texts = batch["texts"]
            va_matrix = batch["va_matrix"]

            audio_mask = None
            if "audio_lengths" in batch:
                max_len = audio.shape[1]
                lengths = batch["audio_lengths"]
                audio_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1).to(self.device)
                audio_mask = audio_mask.float()

            logits = self.model(
                audio_waveform=audio,
                texts=texts,
                audio_attention_mask=audio_mask,
                use_text=use_text,
            )

            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_va_matrices.append(va_matrix)
            all_texts.extend(texts)

        # Concatenate (sample-level, not batch-level)
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_va_matrices = torch.cat(all_va_matrices, dim=0)

        results = {}

        # ── Tier 1: Frame-level ──
        frame_results = compute_frame_metrics(all_logits, all_labels)
        results.update(frame_results)

        # ── Tier 2: Event-level ──
        probs = torch.softmax(all_logits, dim=-1)
        all_gt_events = []
        all_pred_events = []
        all_p_shift = []
        all_gt_shift = []

        for i in range(probs.shape[0]):
            event_data = classify_events(
                probs[i], all_va_matrices[i],
                shift_threshold=self.shift_threshold,
                labels=all_labels[i],
            )
            all_gt_events.append(event_data["gt_events"])
            all_pred_events.append(event_data["pred_events"])
            all_p_shift.append(event_data["p_shift"])
            all_gt_shift.append(event_data["gt_shift"])

        gt_events = np.concatenate(all_gt_events)
        pred_events = np.concatenate(all_pred_events)
        p_shift = np.concatenate(all_p_shift)
        gt_shift = np.concatenate(all_gt_shift)

        event_results = compute_event_metrics(gt_events, pred_events, p_shift, gt_shift)
        results.update(event_results)

        # ── Tier 3: Latency ──
        # Detect shift regions from gt_shift
        shift_regions = self._detect_regions(gt_shift)
        latency_results = compute_eot_latency(p_shift, shift_regions, threshold=self.shift_threshold)
        results.update(latency_results)

        fpr_results = compute_fpr_at_thresholds(p_shift, gt_shift)
        results.update(fpr_results)

        mst_results = compute_mst_fpr_curve(p_shift, gt_shift)
        results["mst_fpr_auc"] = mst_results["mst_fpr_auc"]

        # ── Tier 4: VAQI ──
        hold_regions = self._detect_regions((gt_shift == 0).astype(float))
        vaqi_results = compute_vaqi(
            p_shift, shift_regions, hold_regions,
            threshold=self.shift_threshold,
        )
        results.update(vaqi_results)

        # ── Vietnamese Analysis: Marker Impact ──
        # Build event list with text from batch data for marker analysis
        events_for_analysis = self._build_events_for_analysis(
            gt_events, p_shift, all_va_matrices,
        )
        if events_for_analysis:
            marker_results = analyze_marker_impact(
                events_for_analysis, p_shift, threshold=self.shift_threshold,
            )
            results["marker_analysis"] = marker_results

        return results

    def _detect_regions(self, binary: np.ndarray) -> list:
        """Detect contiguous regions of 1s in a binary array."""
        regions = []
        in_region = False
        start = 0

        for t in range(len(binary)):
            if binary[t] == 1 and not in_region:
                start = t
                in_region = True
            elif binary[t] == 0 and in_region:
                regions.append((start, t))
                in_region = False

        if in_region:
            regions.append((start, len(binary)))

        return regions

    @staticmethod
    def _build_events_for_analysis(
        gt_events: np.ndarray,
        p_shift: np.ndarray,
        all_va_matrices: torch.Tensor,
    ) -> list:
        """
        Build event list for Vietnamese marker impact analysis.

        Extracts shift/hold event boundaries from gt_events and
        creates event dicts with type and eval_frame.
        """
        events = []
        in_event = False
        event_type = None
        start = 0

        for t in range(len(gt_events)):
            current = int(gt_events[t])
            if current in (0, 1) and (not in_event or current != event_type):
                if in_event:
                    mid = (start + t) // 2
                    events.append({
                        "type": "shift" if event_type == 1 else "hold",
                        "eval_frame": mid,
                        "text": "",
                    })
                start = t
                event_type = current
                in_event = True
            elif current == 2:
                if in_event:
                    mid = (start + t) // 2
                    events.append({
                        "type": "shift" if event_type == 1 else "hold",
                        "eval_frame": mid,
                        "text": "",
                    })
                    in_event = False

        if in_event:
            mid = (start + len(gt_events)) // 2
            if mid < len(p_shift):
                events.append({
                    "type": "shift" if event_type == 1 else "hold",
                    "eval_frame": mid,
                    "text": "",
                })

        return events

    def evaluate_with_bootstrap(
        self,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        use_text: bool = True,
    ) -> Dict:
        """
        Evaluate with bootstrap confidence intervals.

        Resamples at the sample (window) level, recomputes metrics each time,
        and returns mean + CI bounds for each scalar metric.

        Args:
            n_bootstrap: Number of bootstrap iterations.
            ci: Confidence level (e.g. 0.95 for 95% CI).
            use_text: Whether to use text modality.

        Returns:
            Dict mapping metric_name -> {"mean": float, "ci_lower": float, "ci_upper": float}
        """
        # Collect all per-sample predictions
        all_logits = []
        all_labels = []
        all_va_matrices = []

        for batch in tqdm(self.test_loader, desc="Collecting predictions"):
            audio = batch["audio_waveform"].to(self.device)
            texts = batch["texts"]

            audio_mask = None
            if "audio_lengths" in batch:
                max_len = audio.shape[1]
                lengths = batch["audio_lengths"]
                audio_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1).to(self.device)
                audio_mask = audio_mask.float()

            with torch.no_grad():
                logits = self.model(
                    audio_waveform=audio,
                    texts=texts,
                    audio_attention_mask=audio_mask,
                    use_text=use_text,
                )

            all_logits.append(logits.cpu())
            all_labels.append(batch["vap_labels"])
            all_va_matrices.append(batch["va_matrix"])

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_va_matrices = torch.cat(all_va_matrices, dim=0)

        N = all_logits.shape[0]

        def _compute_metrics_for_indices(indices):
            sub_logits = all_logits[indices]
            sub_labels = all_labels[indices]
            sub_va = all_va_matrices[indices]

            results = {}
            results.update(compute_frame_metrics(sub_logits, sub_labels))

            probs = torch.softmax(sub_logits, dim=-1)
            gt_ev, pred_ev, p_sh, gt_sh = [], [], [], []
            for i in range(len(indices)):
                ed = classify_events(
                    probs[i], sub_va[i],
                    shift_threshold=self.shift_threshold,
                    labels=sub_labels[i],
                )
                gt_ev.append(ed["gt_events"])
                pred_ev.append(ed["pred_events"])
                p_sh.append(ed["p_shift"])
                gt_sh.append(ed["gt_shift"])

            gt_ev = np.concatenate(gt_ev)
            pred_ev = np.concatenate(pred_ev)
            p_sh = np.concatenate(p_sh)
            gt_sh = np.concatenate(gt_sh)

            results.update(compute_event_metrics(gt_ev, pred_ev, p_sh, gt_sh))
            return results

        # Full evaluation (point estimates)
        full_results = _compute_metrics_for_indices(list(range(N)))

        # Bootstrap resampling
        rng = np.random.RandomState(42)
        alpha = 1.0 - ci
        bootstrap_metrics = {k: [] for k in full_results}

        for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
            boot_idx = rng.choice(N, size=N, replace=True).tolist()
            boot_results = _compute_metrics_for_indices(boot_idx)
            for k, v in boot_results.items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    bootstrap_metrics[k].append(v)

        # Compute CIs
        final = {}
        for k, v in full_results.items():
            samples = bootstrap_metrics.get(k, [])
            if len(samples) >= 10:
                lower = float(np.percentile(samples, 100 * alpha / 2))
                upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
                final[k] = {"mean": float(v), "ci_lower": lower, "ci_upper": upper}
            else:
                final[k] = {"mean": float(v) if isinstance(v, (int, float)) else v,
                             "ci_lower": None, "ci_upper": None}

        return final

    @staticmethod
    def save_report(results: Dict, output_path: str):
        """Save evaluation report to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types
        clean = {}
        for k, v in results.items():
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
            elif isinstance(v, dict):
                clean[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                            for kk, vv in v.items()}
            else:
                clean[k] = v

        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"Evaluation report saved to {path}")

    @staticmethod
    def print_report(results: Dict):
        """Pretty-print evaluation results."""
        print("\n" + "=" * 60)
        print("MM-VAP-VI Evaluation Report")
        print("=" * 60)

        # Tier 1
        print("\n--- Tier 1: Frame-Level ---")
        for key in ["frame_ce", "frame_perplexity", "frame_top1_acc", "frame_top5_acc",
                     "frame_weighted_f1", "frame_ece", "frame_brier"]:
            if key in results:
                val = results[key]
                if isinstance(val, dict):
                    val = val["mean"]
                print(f"  {key}: {val:.4f}")

        # Tier 2
        print("\n--- Tier 2: Event-Level ---")
        for key in ["shift_hold_ba", "bc_f1", "predict_shift_auc", "event_3class_ba"]:
            if key in results:
                val = results[key]
                if isinstance(val, dict):
                    val = val["mean"]
                print(f"  {key}: {val:.4f}")

        # Tier 3
        print("\n--- Tier 3: Latency ---")
        for key in ["eot_latency_mean_ms", "eot_latency_median_ms", "detection_rate", "mst_fpr_auc"]:
            if key in results:
                val = results[key]
                if isinstance(val, dict):
                    val = val["mean"]
                print(f"  {key}: {val:.4f}")

        # FPR at thresholds
        fpr_keys = [k for k in results if k.startswith("fpr_at_")]
        if fpr_keys:
            print("\n  FPR at thresholds:")
            for key in sorted(fpr_keys):
                val = results[key]
                if isinstance(val, dict):
                    val = val["mean"]
                print(f"    {key}: {val:.4f}")

        # Tier 4
        if "vaqi" in results:
            print("\n--- Tier 4: VAQI ---")
            for key in ["vaqi", "vaqi_interruption_rate", "vaqi_missed_response_rate",
                         "vaqi_latency_score", "vaqi_median_latency_ms"]:
                if key in results:
                    val = results[key]
                    if isinstance(val, dict):
                        val = val["mean"]
                    print(f"  {key}: {val:.4f}" if key != "vaqi" else f"  {key}: {val:.1f}/100")

        # Vietnamese marker analysis
        if "marker_analysis" in results:
            ma = results["marker_analysis"]
            print("\n--- Vietnamese Marker Analysis ---")
            wm = ma.get("with_marker", {})
            wo = ma.get("without_marker", {})
            mb = ma.get("marker_benefit", {})

            def _fmt(v, signed=False):
                if isinstance(v, (int, float)) and not np.isnan(v):
                    return f"{v:+.4f}" if signed else f"{v:.4f}"
                return "N/A"

            wm_n = wm.get("shift_count", 0) + wm.get("hold_count", 0)
            wo_n = wo.get("shift_count", 0) + wo.get("hold_count", 0)
            print(f"  With marker:    shift_acc={_fmt(wm.get('shift_accuracy'))}, "
                  f"hold_acc={_fmt(wm.get('hold_accuracy'))} (n={wm_n})")
            print(f"  Without marker: shift_acc={_fmt(wo.get('shift_accuracy'))}, "
                  f"hold_acc={_fmt(wo.get('hold_accuracy'))} (n={wo_n})")
            print(f"  Marker benefit: shift={_fmt(mb.get('shift_delta'), signed=True)}, "
                  f"hold={_fmt(mb.get('hold_delta'), signed=True)}")

        print("=" * 60)
