#!/usr/bin/env python3
"""
Visualize model predictions vs ground truth for qualitative analysis.

Generates timeline plots showing:
- Speaker voice activity (ground truth)
- P(shift) prediction curve
- Ground truth events (shift/hold/BC)
- Predicted events
- Text overlay with discourse markers highlighted

Usage:
    python scripts/visualize_predictions.py \
        --checkpoint outputs/mm_vap/best_model.pt \
        --num-windows 5 \
        --output outputs/figures/
"""

import sys
import argparse
import json
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")

from src.models.model import MMVAPModel
from src.data.dataset import VAPDataset
from src.data.collate import vap_collate_fn
from src.evaluation.event_metrics import classify_events, probs_to_p_now
from src.evaluation.vietnamese_analysis import DEFAULT_MARKER_SETS
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize VAP predictions")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="configs/config.yaml", help="Config YAML")
    parser.add_argument("--test-manifest", default=None, help="Test manifest JSON")
    parser.add_argument("--num-windows", type=int, default=5, help="Number of windows to plot")
    parser.add_argument("--window-indices", type=int, nargs="*", default=None,
                        help="Specific window indices to plot")
    parser.add_argument("--output", default="outputs/figures", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--no-text", action="store_true", help="Audio-only mode")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    return parser.parse_args()


def plot_window(
    ax_list,
    va_matrix: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    gt_events: np.ndarray,
    pred_events: np.ndarray,
    text: str,
    frame_hz: int = 50,
    title: str = "",
):
    """Plot a single window with 4 subplots."""
    T = len(gt_events)
    time_sec = np.arange(T) / frame_hz

    ax_va, ax_prob, ax_gt, ax_pred = ax_list

    # --- Panel 1: Voice Activity ---
    ax_va.fill_between(time_sec, 0, va_matrix[0, :T], alpha=0.5, color="tab:blue", label="Speaker 0")
    ax_va.fill_between(time_sec, 0, -va_matrix[1, :T], alpha=0.5, color="tab:orange", label="Speaker 1")
    ax_va.set_ylim(-1.3, 1.3)
    ax_va.set_ylabel("Voice Activity")
    ax_va.legend(loc="upper right", fontsize=7)
    ax_va.set_title(title, fontsize=10, fontweight="bold")
    ax_va.set_xticklabels([])

    # --- Panel 2: P(shift) probability ---
    ax_prob.plot(time_sec, p0, color="tab:blue", alpha=0.7, linewidth=0.8, label="P(s0 active)")
    ax_prob.plot(time_sec, p1, color="tab:orange", alpha=0.7, linewidth=0.8, label="P(s1 active)")
    ax_prob.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_ylabel("P(active)")
    ax_prob.legend(loc="upper right", fontsize=7)
    ax_prob.set_xticklabels([])

    # --- Panel 3: GT events ---
    event_colors = {0: "lightgray", 1: "tab:red", 2: "tab:green"}
    event_names = {0: "Hold", 1: "Shift", 2: "BC"}
    for t_idx in range(T - 1):
        ax_gt.axvspan(time_sec[t_idx], time_sec[t_idx + 1],
                      color=event_colors[gt_events[t_idx]], alpha=0.7)
    ax_gt.set_ylabel("GT Events")
    ax_gt.set_ylim(0, 1)
    ax_gt.set_yticks([])
    ax_gt.set_xticklabels([])

    # --- Panel 4: Predicted events ---
    for t_idx in range(T - 1):
        ax_pred.axvspan(time_sec[t_idx], time_sec[t_idx + 1],
                        color=event_colors[pred_events[t_idx]], alpha=0.7)
    ax_pred.set_ylabel("Pred Events")
    ax_pred.set_ylim(0, 1)
    ax_pred.set_yticks([])
    ax_pred.set_xlabel("Time (seconds)")

    # Legend for events
    patches = [mpatches.Patch(color=c, label=event_names[k]) for k, c in event_colors.items()]
    ax_pred.legend(handles=patches, loc="upper right", fontsize=7, ncol=3)

    # Add text annotation with markers highlighted
    all_markers = [m for markers in DEFAULT_MARKER_SETS.values() for m in markers]
    if text:
        # Truncate for display
        display_text = text[-120:] if len(text) > 120 else text
        # Bold markers
        for marker in all_markers:
            if marker in display_text:
                display_text = display_text.replace(marker, f"[{marker}]")
        ax_va.text(0.01, 0.95, f"Text: {display_text}", transform=ax_va.transAxes,
                   fontsize=6, verticalalignment="top",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))


def main():
    if not HAS_MPL:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load model
    print("Loading model...")
    model = MMVAPModel.from_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Build dataset
    data_cfg = config["data"]
    manifest_dir = Path(data_cfg.get("manifest_dir", "data"))
    test_manifest = args.test_manifest or str(manifest_dir / "vap_manifest_test.json")
    frame_hz = data_cfg.get("frame_hz", 50)

    test_dataset = VAPDataset(
        manifest_path=test_manifest,
        window_sec=data_cfg.get("window_sec", 20.0),
        stride_sec=data_cfg.get("stride_sec", 5.0),
        frame_hz=frame_hz,
        sample_rate=data_cfg.get("sample_rate", 16000),
    )

    # Select windows
    if args.window_indices:
        indices = args.window_indices
    else:
        # Pick evenly spaced windows
        total = len(test_dataset)
        step = max(1, total // args.num_windows)
        indices = list(range(0, total, step))[:args.num_windows]

    print(f"Plotting {len(indices)} windows: {indices}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_text = not args.no_text
    shift_threshold = config.get("evaluation", {}).get("shift_threshold", 0.5)

    for idx in indices:
        sample = test_dataset[idx]
        audio = sample["audio_waveform"].unsqueeze(0).to(device)
        labels = sample["vap_labels"]
        va_matrix = sample["va_matrix"]
        text = sample["text"]
        file_id = sample["file_id"]
        win_start = sample["window_start_frame"]

        with torch.no_grad():
            logits = model(
                audio_waveform=audio,
                texts=[text],
                use_text=use_text,
            )

        probs = torch.softmax(logits[0].cpu(), dim=-1)
        p0, p1 = probs_to_p_now(probs.unsqueeze(0))
        p0, p1 = p0[0].numpy(), p1[0].numpy()

        # Frame alignment
        T_min = min(probs.shape[0], labels.shape[0], va_matrix.shape[1])
        probs = probs[:T_min]
        labels_aligned = labels[:T_min]
        va_aligned = va_matrix[:, :T_min].numpy()
        p0 = p0[:T_min]
        p1 = p1[:T_min]

        event_data = classify_events(
            probs, va_matrix[:, :T_min],
            shift_threshold=shift_threshold,
            labels=labels_aligned,
        )

        fig, axes = plt.subplots(4, 1, figsize=(16, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 2, 1, 1]})
        fig.subplots_adjust(hspace=0.15)

        title = f"{file_id} | Window {idx} (frame {win_start}–{win_start + T_min})"
        plot_window(
            axes, va_aligned, p0, p1,
            event_data["gt_events"], event_data["pred_events"],
            text=text, frame_hz=frame_hz, title=title,
        )

        out_path = output_dir / f"window_{idx:04d}_{file_id}.png"
        fig.savefig(str(out_path), dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # --- Summary confusion matrix ---
    print("\nGenerating confusion matrix over all test data...")
    loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False,
        num_workers=0, collate_fn=vap_collate_fn,
    )

    all_gt, all_pred = [], []
    with torch.no_grad():
        for batch in loader:
            audio_b = batch["audio_waveform"].to(device)
            texts_b = batch["texts"]
            labels_b = batch["vap_labels"]
            va_b = batch["va_matrix"]

            audio_mask = None
            if "audio_lengths" in batch:
                max_len = audio_b.shape[1]
                lengths = batch["audio_lengths"]
                audio_mask = (torch.arange(max_len, device=device).unsqueeze(0) <
                              lengths.unsqueeze(1).to(device)).float()

            logits_b = model(
                audio_waveform=audio_b, texts=texts_b,
                audio_attention_mask=audio_mask, use_text=use_text,
            )
            probs_b = torch.softmax(logits_b.cpu(), dim=-1)

            for i in range(probs_b.shape[0]):
                T_min = min(probs_b[i].shape[0], labels_b[i].shape[0], va_b[i].shape[1])
                ed = classify_events(
                    probs_b[i, :T_min], va_b[i, :, :T_min],
                    shift_threshold=shift_threshold,
                    labels=labels_b[i, :T_min],
                )
                all_gt.append(ed["gt_events"])
                all_pred.append(ed["pred_events"])

    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    labels_list = [0, 1, 2]
    display_labels = ["Hold", "Shift", "BC"]
    cm = confusion_matrix(all_gt, all_pred, labels=labels_list)

    fig_cm, ax_cm = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    disp1 = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp1.plot(ax=ax_cm[0], cmap="Blues", values_format="d")
    ax_cm[0].set_title("Confusion Matrix (counts)")

    # Normalized
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=display_labels)
    disp2.plot(ax=ax_cm[1], cmap="Blues", values_format=".2f")
    ax_cm[1].set_title("Confusion Matrix (normalized)")

    cm_path = output_dir / "confusion_matrix.png"
    fig_cm.savefig(str(cm_path), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_cm)
    print(f"  Saved: {cm_path}")

    # --- Per-speaker analysis ---
    print("\nPer-speaker analysis...")
    # Speaker 0 = frames where s0 is dominant, Speaker 1 = frames where s1 is dominant
    all_s0_acc, all_s1_acc = [], []
    all_s0_gt, all_s1_gt, all_s0_pred, all_s1_pred = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            audio_b = batch["audio_waveform"].to(device)
            texts_b = batch["texts"]
            labels_b = batch["vap_labels"]
            va_b = batch["va_matrix"]

            audio_mask = None
            if "audio_lengths" in batch:
                max_len = audio_b.shape[1]
                lengths = batch["audio_lengths"]
                audio_mask = (torch.arange(max_len, device=device).unsqueeze(0) <
                              lengths.unsqueeze(1).to(device)).float()

            logits_b = model(
                audio_waveform=audio_b, texts=texts_b,
                audio_attention_mask=audio_mask, use_text=use_text,
            )

            for i in range(logits_b.shape[0]):
                T_min = min(logits_b[i].shape[0], labels_b[i].shape[0], va_b[i].shape[1])
                preds = logits_b[i, :T_min].cpu().argmax(dim=-1)
                labs = labels_b[i, :T_min]
                va = va_b[i, :, :T_min].numpy()

                valid = labs >= 0
                if not valid.any():
                    continue

                # s0-dominant frames: s0 active, s1 silent
                s0_dom = (va[0] == 1) & (va[1] == 0) & valid.numpy()
                s1_dom = (va[1] == 1) & (va[0] == 0) & valid.numpy()

                if s0_dom.sum() > 0:
                    acc = (preds.numpy()[s0_dom] == labs.numpy()[s0_dom]).mean()
                    all_s0_acc.append(float(acc))
                if s1_dom.sum() > 0:
                    acc = (preds.numpy()[s1_dom] == labs.numpy()[s1_dom]).mean()
                    all_s1_acc.append(float(acc))

    s0_mean = np.mean(all_s0_acc) if all_s0_acc else float("nan")
    s1_mean = np.mean(all_s1_acc) if all_s1_acc else float("nan")

    speaker_stats = {
        "speaker_0_mean_acc": round(float(s0_mean), 4),
        "speaker_0_n_windows": len(all_s0_acc),
        "speaker_1_mean_acc": round(float(s1_mean), 4),
        "speaker_1_n_windows": len(all_s1_acc),
        "speaker_bias": round(float(abs(s0_mean - s1_mean)), 4),
    }

    # Save per-speaker stats
    stats_path = output_dir / "per_speaker_analysis.json"
    with open(str(stats_path), "w") as f:
        json.dump(speaker_stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    print(f"\n  Speaker 0 mean acc: {s0_mean:.4f} (n={len(all_s0_acc)})")
    print(f"  Speaker 1 mean acc: {s1_mean:.4f} (n={len(all_s1_acc)})")
    print(f"  Speaker bias (|diff|): {abs(s0_mean - s1_mean):.4f}")

    # Save confusion matrix data as JSON too
    cm_data = {
        "labels": display_labels,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }
    cm_json_path = output_dir / "confusion_matrix.json"
    with open(str(cm_json_path), "w") as f:
        json.dump(cm_data, f, indent=2)
    print(f"  Saved: {cm_json_path}")

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
