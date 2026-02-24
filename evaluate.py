#!/usr/bin/env python3
"""
evaluate.py - Evaluation entry point for MM-VAP-VI.

Runs trained model on test set and computes all metrics:
- Tier 1: Frame-level (accuracy, F1, perplexity, ECE)
- Tier 2: Event-level (shift/hold BA, BC F1, shift AUC)
- Tier 3: Latency (EoT latency, FPR curves, VAQI)
- Vietnamese marker analysis

Usage:
    # Basic evaluation
    python evaluate.py --checkpoint outputs/mm_vap/best_model.pt

    # With bootstrap confidence intervals
    python evaluate.py --checkpoint outputs/mm_vap/best_model.pt --bootstrap 1000

    # Audio-only evaluation (no text)
    python evaluate.py --checkpoint outputs/mm_vap/best_model.pt --no-text

    # Compare two models
    python evaluate.py \
        --checkpoint outputs/mm_vap/best_model.pt \
        --checkpoint-b outputs/ablation_no_hutu/best_model.pt \
        --compare
"""

import sys
import argparse
import json
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.models.model import MMVAPModel
from src.data.dataset import VAPDataset
from src.data.collate import vap_collate_fn
from src.evaluation.evaluator import MMVAPEvaluator
from src.evaluation.statistical import permutation_test
from src.evaluation.frame_metrics import compute_frame_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MM-VAP-VI model")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--checkpoint-b", default=None,
        help="Path to second model checkpoint for comparison",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--test-manifest", default=None,
        help="Path to test manifest JSON (default: data/vap_manifest_test.json)",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=0,
        help="Number of bootstrap iterations for CI (0 = no bootstrap)",
    )
    parser.add_argument(
        "--no-text", action="store_true",
        help="Evaluate in audio-only mode (no text)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare two models with permutation test",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device (cuda, cpu)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader num_workers",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for JSON report",
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, device: str):
    """Load model from checkpoint."""
    model = MMVAPModel.from_config(config_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    info = {
        "stage": ckpt.get("stage", "?"),
        "epoch": ckpt.get("epoch", "?"),
        "global_step": ckpt.get("global_step", "?"),
        "best_metric": ckpt.get("best_metric", "?"),
    }
    return model, info


def build_test_loader(config: dict, args) -> DataLoader:
    """Build test DataLoader."""
    data_cfg = config["data"]
    manifest_dir = Path(data_cfg.get("manifest_dir", "data"))
    test_manifest = args.test_manifest or str(manifest_dir / "vap_manifest_test.json")

    if not Path(test_manifest).exists():
        print(f"ERROR: Test manifest not found: {test_manifest}")
        sys.exit(1)

    test_dataset = VAPDataset(
        manifest_path=test_manifest,
        window_sec=data_cfg.get("window_sec", 20.0),
        stride_sec=data_cfg.get("stride_sec", 5.0),
        frame_hz=data_cfg.get("frame_hz", 50),
        sample_rate=data_cfg.get("sample_rate", 16000),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=vap_collate_fn,
        pin_memory=True,
    )

    return test_loader


def evaluate_single(args):
    """Evaluate a single model."""
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("MM-VAP-VI Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Config: {args.config}")
    print(f"  Device: {args.device}")
    print(f"  Use text: {not args.no_text}")
    if args.bootstrap > 0:
        print(f"  Bootstrap: {args.bootstrap} iterations")
    print()

    # Load model
    print("Loading model...")
    model, ckpt_info = load_model_from_checkpoint(args.checkpoint, args.config, args.device)
    print(f"  Checkpoint: stage={ckpt_info['stage']}, epoch={ckpt_info['epoch']}, "
          f"best_metric={ckpt_info['best_metric']}")

    # Build test loader
    print("Building test loader...")
    test_loader = build_test_loader(config, args)
    print(f"  Test: {len(test_loader.dataset)} windows, {len(test_loader)} batches")
    print()

    # Evaluate
    evaluator = MMVAPEvaluator(
        model=model,
        test_loader=test_loader,
        device=args.device,
        shift_threshold=config.get("evaluation", {}).get("shift_threshold", 0.5),
        bc_threshold=config.get("evaluation", {}).get("bc_threshold", 0.3),
    )

    use_text = not args.no_text

    if args.bootstrap > 0:
        print(f"Running evaluation with {args.bootstrap} bootstrap iterations...")
        results = evaluator.evaluate_with_bootstrap(
            n_bootstrap=args.bootstrap,
            ci=config.get("evaluation", {}).get("bootstrap_ci", 0.95),
            use_text=use_text,
        )
    else:
        print("Running evaluation...")
        results = evaluator.evaluate(use_text=use_text)

    # Print report
    evaluator.print_report(results)

    # Save report
    output_path = args.output or f"outputs/eval_{'bootstrap_' if args.bootstrap > 0 else ''}report.json"
    evaluator.save_report(results, output_path)

    return results


def evaluate_compare(args):
    """Compare two models with permutation test."""
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("MM-VAP-VI Model Comparison")
    print("=" * 60)
    print(f"  Model A: {args.checkpoint}")
    print(f"  Model B: {args.checkpoint_b}")
    print()

    # Build test loader (shared)
    test_loader = build_test_loader(config, args)
    print(f"  Test: {len(test_loader.dataset)} windows")
    print()

    use_text = not args.no_text

    # Evaluate Model A
    print("--- Model A ---")
    model_a, info_a = load_model_from_checkpoint(args.checkpoint, args.config, args.device)
    eval_a = MMVAPEvaluator(model_a, test_loader, device=args.device)
    results_a = eval_a.evaluate(use_text=use_text)
    eval_a.print_report(results_a)

    # Evaluate Model B
    print("\n--- Model B ---")
    model_b, info_b = load_model_from_checkpoint(args.checkpoint_b, args.config, args.device)
    eval_b = MMVAPEvaluator(model_b, test_loader, device=args.device)
    results_b = eval_b.evaluate(use_text=use_text)
    eval_b.print_report(results_b)

    # Per-sample scores for permutation test
    # Collect per-window accuracy for both models
    print("\nComputing per-window scores for permutation test...")

    def get_per_window_accuracy(model, loader):
        """Get accuracy per window for permutation test."""
        model.to(args.device)
        model.eval()
        scores = []

        with torch.no_grad():
            for batch in loader:
                audio = batch["audio_waveform"].to(args.device)
                labels = batch["vap_labels"]
                texts = batch["texts"]

                audio_mask = None
                if "audio_lengths" in batch:
                    max_len = audio.shape[1]
                    lengths = batch["audio_lengths"]
                    audio_mask = torch.arange(max_len, device=args.device).unsqueeze(0) < lengths.unsqueeze(1).to(args.device)
                    audio_mask = audio_mask.float()

                logits = model(
                    audio_waveform=audio, texts=texts,
                    audio_attention_mask=audio_mask, use_text=use_text,
                )

                # Align frames
                T_logits = logits.shape[1]
                T_labels = labels.shape[1]
                if T_logits < T_labels:
                    labels = labels[:, :T_logits]
                elif T_logits > T_labels:
                    logits = logits[:, :T_labels, :]

                preds = logits.cpu().argmax(dim=-1)
                for i in range(preds.shape[0]):
                    valid = labels[i] >= 0
                    if valid.any():
                        acc = (preds[i][valid] == labels[i][valid]).float().mean().item()
                        scores.append(acc)

        return np.array(scores)

    scores_a = get_per_window_accuracy(model_a, test_loader)
    scores_b = get_per_window_accuracy(model_b, test_loader)

    # Permutation test
    perm_result = permutation_test(scores_a, scores_b, n_permutations=10000)

    print(f"\n{'='*60}")
    print("Permutation Test Results")
    print(f"{'='*60}")
    print(f"  Model A mean acc: {scores_a.mean():.4f}")
    print(f"  Model B mean acc: {scores_b.mean():.4f}")
    print(f"  Observed diff (B-A): {perm_result['observed_diff']:.4f}")
    print(f"  p-value: {perm_result['p_value']:.4f}")
    print(f"  Significant at 0.05: {perm_result['significant_at_05']}")
    print(f"  Significant at 0.01: {perm_result['significant_at_01']}")
    print(f"  N windows: {perm_result['n_conversations']}")

    # Cohen's d
    pooled_std = np.sqrt(
        ((len(scores_a)-1)*np.var(scores_a, ddof=1) +
         (len(scores_b)-1)*np.var(scores_b, ddof=1)) /
        (len(scores_a) + len(scores_b) - 2)
    )
    cohens_d = (scores_b.mean() - scores_a.mean()) / pooled_std if pooled_std > 0 else 0
    effect = "negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
    print(f"  Cohen's d: {cohens_d:.3f} ({effect})")
    print(f"{'='*60}")

    # Save comparison report
    comparison = {
        "model_a": {"path": args.checkpoint, "results": results_a, "mean_acc": float(scores_a.mean())},
        "model_b": {"path": args.checkpoint_b, "results": results_b, "mean_acc": float(scores_b.mean())},
        "permutation_test": perm_result,
        "cohens_d": float(cohens_d),
        "effect_size": effect,
    }

    output_path = args.output or "outputs/comparison_report.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nComparison report saved to {output_path}")


def main():
    args = parse_args()

    if args.compare and args.checkpoint_b:
        evaluate_compare(args)
    else:
        evaluate_single(args)


if __name__ == "__main__":
    main()
