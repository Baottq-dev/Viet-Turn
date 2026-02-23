#!/usr/bin/env python3
"""
train.py - Training entry point for MM-VAP-VI.

Usage:
    # Full 3-stage training
    python train.py --config configs/config.yaml

    # Resume from checkpoint
    python train.py --config configs/config.yaml --resume outputs/mm_vap/checkpoint_s2_e5.pt

    # Override output directory
    python train.py --config configs/config.yaml --output-dir outputs/experiment_1

    # Specify device
    python train.py --config configs/config.yaml --device cuda:1
"""

import sys
import argparse
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import DataLoader

from src.models.model import MMVAPModel
from src.data.dataset import VAPDataset
from src.data.collate import vap_collate_fn
from src.training.trainer import VAPTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train MM-VAP-VI model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to model/training config YAML",
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default=None,
        help="Path to train manifest JSON (overrides config)",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=None,
        help="Path to val manifest JSON (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader num_workers",
    )
    return parser.parse_args()


def build_dataloaders(config: dict, args) -> tuple:
    """Build train and val DataLoaders from config."""
    data_cfg = config["data"]
    train_cfg = config["training"]

    # Resolve manifest paths
    manifest_dir = Path(data_cfg.get("manifest_dir", "data"))
    train_manifest = args.train_manifest or str(manifest_dir / "vap_manifest_train.json")
    val_manifest = args.val_manifest or str(manifest_dir / "vap_manifest_val.json")

    # Build datasets
    train_dataset = VAPDataset(
        manifest_path=train_manifest,
        window_sec=data_cfg.get("window_sec", 20.0),
        stride_sec=data_cfg.get("stride_sec", 5.0),
        frame_hz=data_cfg.get("frame_hz", 50),
        sample_rate=data_cfg.get("sample_rate", 16000),
    )

    val_dataset = None
    if Path(val_manifest).exists():
        val_dataset = VAPDataset(
            manifest_path=val_manifest,
            window_sec=data_cfg.get("window_sec", 20.0),
            stride_sec=data_cfg.get("stride_sec", 5.0),
            frame_hz=data_cfg.get("frame_hz", 50),
            sample_rate=data_cfg.get("sample_rate", 16000),
        )

    batch_size = train_cfg.get("batch_size", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=vap_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=vap_collate_fn,
            pin_memory=True,
        )

    return train_loader, val_loader


def main():
    args = parse_args()

    print("=" * 60)
    print("MM-VAP-VI Training")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Device: {args.device}")
    if args.resume:
        print(f"  Resume: {args.resume}")
    print()

    # Build model
    print("Building model...")
    model = MMVAPModel.from_config(args.config)
    params = model.count_parameters()
    print(f"  Total params: {params['total']['total']:,}")
    print(f"  Trainable params: {params['total']['trainable']:,}")
    print()

    # Build dataloaders
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(config, args)
    print(f"  Train: {len(train_loader.dataset)} windows, {len(train_loader)} batches")
    if val_loader:
        print(f"  Val: {len(val_loader.dataset)} windows, {len(val_loader)} batches")
    print()

    # Output directory
    output_dir = args.output_dir or config.get("logging", {}).get("output_dir", "outputs/mm_vap")

    # Build trainer
    trainer = VAPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config_path=args.config,
        device=args.device,
        output_dir=output_dir,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)
        print(f"  Resumed at stage {trainer.resume_stage}, epoch {trainer.resume_epoch}")
        print(f"  Global step: {trainer.global_step}")
        print(f"  Best metric: {trainer.best_metric:.4f}")
        print()

    # Train
    best_metric = trainer.train()

    print(f"\nTraining finished. Best {trainer.es_metric}: {best_metric:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
