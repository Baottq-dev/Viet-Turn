#!/usr/bin/env python3
"""
train.py - Main training script for VietTurnEdge model

Connects: Dataset -> DataLoader -> Model -> Trainer -> Evaluation

Usage:
    python train.py
    python train.py --epochs 50 --batch-size 16 --lr 1e-4
    python train.py --resume checkpoints/best_model.pt
    python train.py --device cuda
"""

# Fix OpenMP conflict on Windows
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

# Local imports
from src.data.audio_processor import AudioProcessor
from src.data.dataset import TurnTakingDataset, create_dataloader
from src.models.viet_turn_edge import VietTurnEdge
from src.training.trainer import Trainer
from src.training.losses import FocalLoss


def get_class_weights(train_path: str) -> torch.Tensor:
    """Calculate class weights from training data for imbalanced handling."""
    with open(train_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    
    counts = {"YIELD": 0, "HOLD": 0, "BACKCHANNEL": 0}
    for seg in segments:
        label = seg.get("label", "YIELD").upper()
        if label in counts:
            counts[label] += 1
    
    total = sum(counts.values())
    # Inverse frequency weighting
    weights = [
        total / (3 * counts["YIELD"]) if counts["YIELD"] > 0 else 1.0,
        total / (3 * counts["HOLD"]) if counts["HOLD"] > 0 else 1.0,
        total / (3 * counts["BACKCHANNEL"]) if counts["BACKCHANNEL"] > 0 else 1.0,
    ]
    
    print(f"📊 Class distribution: {counts}")
    print(f"📊 Class weights: YIELD={weights[0]:.2f}, HOLD={weights[1]:.2f}, BACKCHANNEL={weights[2]:.2f}")
    
    return torch.FloatTensor(weights)


def main():
    parser = argparse.ArgumentParser(
        description="Train VietTurnEdge model for Vietnamese turn-taking prediction"
    )
    
    # Data args
    parser.add_argument("--train-data", default="datasets/final/train.json",
                        help="Path to training data")
    parser.add_argument("--val-data", default="datasets/final/val.json",
                        help="Path to validation data")
    parser.add_argument("--audio-dir", default="datasets/raw/youtube",
                        help="Directory containing audio files")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    
    # Model args
    parser.add_argument("--acoustic-dim", type=int, default=128,
                        help="Acoustic branch hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    
    # Loss args
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter")
    parser.add_argument("--use-class-weights", action="store_true", default=True,
                        help="Use class weights for imbalanced data")
    
    # Device
    parser.add_argument("--device", default="cuda",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to use")
    
    # Checkpointing
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    
    # Misc
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of data loader workers (0 for Windows)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"\n{'='*60}")
    print(f"  VietTurnEdge Training")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"{'='*60}\n")
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Check data exists
    if not Path(args.train_data).exists():
        print(f"❌ Training data not found: {args.train_data}")
        print("   Run: python scripts/06_split_dataset.py first")
        sys.exit(1)
    
    # Initialize audio processor
    print("🎵 Initializing audio processor...")
    audio_processor = AudioProcessor(
        sample_rate=16000,
        n_mels=40,
        frame_shift_ms=10
    )
    
    # Create data loaders
    print("📂 Loading datasets...")
    train_loader = create_dataloader(
        data_path=args.train_data,
        audio_processor=audio_processor,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        use_precut=False,  # Cut on-the-fly from original audio
        audio_dir=args.audio_dir
    )
    
    val_loader = None
    if Path(args.val_data).exists():
        val_loader = create_dataloader(
            data_path=args.val_data,
            audio_processor=audio_processor,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            use_precut=False,
            audio_dir=args.audio_dir
        )
    
    # Calculate class weights
    class_weights = None
    if args.use_class_weights:
        class_weights = get_class_weights(args.train_data)
    
    # Build model
    print("🧠 Building model...")
    model_config = {
        "acoustic_branch": {
            "input_dim": 42,  # 40 mel + 1 f0 + 1 energy
            "hidden_dim": args.acoustic_dim,
            "output_dim": 64,
            "num_layers": 4,
            "kernel_size": 3,
            "dropout": args.dropout
        },
        "linguistic_branch": {
            "pretrained": "vinai/phobert-base-v2",
            "output_dim": 64,
            "freeze_embeddings": True,
            "use_marker_detection": True
        },
        "fusion": {
            "acoustic_dim": 64,
            "linguistic_dim": 64,
            "hidden_dim": 128,
            "output_dim": 64
        },
        "model": {
            "num_classes": 3
        }
    }
    
    model = VietTurnEdge.from_config(model_config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Build loss function
    criterion = FocalLoss(
        gamma=args.focal_gamma,
        alpha=class_weights.tolist() if class_weights is not None else None
    )
    
    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Build scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Training config
    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "focal_gamma": args.focal_gamma,
        "device": device,
        "model_config": model_config,
    }
    
    # Create trainer
    print("🏋️ Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=training_config,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.wandb
    )
    
    # Resume if specified
    if args.resume:
        print(f"📂 Resuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)
    
    # Train!
    print("\n🚀 Starting training...")
    start_time = datetime.now()
    
    try:
        trainer.train(
            epochs=args.epochs,
            save_every=args.save_every
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        trainer.save_checkpoint("interrupted.pt")
        print("   Saved checkpoint: interrupted.pt")
    
    elapsed = datetime.now() - start_time
    print(f"\n✅ Training completed in {elapsed}")
    print(f"   Best model saved to: {args.checkpoint_dir}/best_model.pt")
    
    # Final evaluation on test set if available
    test_data = "datasets/final/test.json"
    if Path(test_data).exists():
        print("\n📊 Evaluating on test set...")
        test_loader = create_dataloader(
            data_path=test_data,
            audio_processor=audio_processor,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            use_precut=False,
            audio_dir=args.audio_dir
        )
        
        # Load best model
        trainer.load_checkpoint(f"{args.checkpoint_dir}/best_model.pt")
        test_metrics = trainer.validate()
        
        print(f"\n📈 Test Results:")
        print(f"   Accuracy: {test_metrics['val_acc']:.4f}")
        print(f"   F1 Score: {test_metrics['val_f1']:.4f}")
        print(f"   Loss: {test_metrics['val_loss']:.4f}")


if __name__ == "__main__":
    main()
