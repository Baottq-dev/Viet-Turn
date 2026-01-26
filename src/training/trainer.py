"""
Trainer class for Viet-TurnEdge model training.
Handles training loop, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Dict, Optional, Callable
import time
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Trainer:
    """
    Training handler for VietTurnEdge model.
    
    Usage:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config
        )
        trainer.train(epochs=50)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        device: str = "cuda",
        config: Optional[Dict] = None,
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or {}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        if criterion is None:
            from .losses import FocalLoss
            self.criterion = FocalLoss(gamma=2.0)
        else:
            self.criterion = criterion
        
        # Optimizer
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-5)
        
        if optimizer is None:
            self.optimizer = AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Scheduler
        if scheduler is None:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get("epochs", 50),
                eta_min=self.config.get("min_lr", 1e-6)
            )
        else:
            self.scheduler = scheduler
        
        # Tracking
        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.current_epoch = 0
        
        # Wandb
        self.use_wandb = use_wandb and HAS_WANDB
        if self.use_wandb:
            wandb.init(
                project=self.config.get("wandb_project", "viet-turn-edge"),
                config=self.config
            )
    
    def train_epoch(self) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if len(batch) == 3:
                features, labels, metadata = batch
            else:
                features, labels = batch
                metadata = None
            
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(features)
            
            # Get logits from output dict if needed
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs
            
            # Handle temporal dimension: take last frame for segment-level prediction
            if logits.dim() == 3:
                # logits: (B, T, num_classes) -> take last frame -> (B, num_classes)
                logits = logits[:, -1, :]
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            grad_clip = self.config.get("gradient_clip", 1.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            self.optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            
            # Accuracy
            if logits.dim() == 3:
                preds = logits[:, -1, :].argmax(dim=-1)
            else:
                preds = logits.argmax(dim=-1)
            
            if labels.dim() == 2:
                labels_flat = labels[:, -1]
            else:
                labels_flat = labels
            
            correct += (preds == labels_flat).sum().item()
            total += labels_flat.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct/total:.4f}"
            })
        
        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "train_acc": correct / total
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict:
        """Validate on validation set."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            if len(batch) == 3:
                features, labels, metadata = batch
            else:
                features, labels = batch
            
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(features)
            
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs
            
            # Handle temporal dimension: take last frame
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            
            # Predictions
            preds = logits.argmax(dim=-1)
            
            if labels.dim() == 2:
                labels_flat = labels[:, -1]
            else:
                labels_flat = labels
            
            correct += (preds == labels_flat).sum().item()
            total += labels_flat.size(0)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels_flat.cpu().tolist())
        
        # Compute F1
        f1 = self._compute_f1(all_preds, all_labels)
        
        metrics = {
            "val_loss": total_loss / len(self.val_loader),
            "val_acc": correct / total,
            "val_f1": f1
        }
        
        return metrics
    
    def _compute_f1(self, preds, labels):
        """Compute macro F1 score."""
        num_classes = 3
        f1_scores = []
        
        for c in range(num_classes):
            tp = sum(1 for p, l in zip(preds, labels) if p == c and l == c)
            fp = sum(1 for p, l in zip(preds, labels) if p == c and l != c)
            fn = sum(1 for p, l in zip(preds, labels) if p != c and l == c)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        return sum(f1_scores) / len(f1_scores)
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "best_val_f1": self.best_val_f1,
            "config": self.config
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_val_f1 = checkpoint.get("best_val_f1", 0.0)
    
    def train(self, epochs: int = 50, save_every: int = 5):
        """Full training loop."""
        print(f"🚀 Starting training for {epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"   Val batches: {len(self.val_loader)}")
        
        patience = self.config.get("patience", 10)
        no_improve = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate() if self.val_loader else {}
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            all_metrics = {**train_metrics, **val_metrics}
            print(f"\nEpoch {self.current_epoch}: {all_metrics}")
            
            if self.use_wandb:
                wandb.log(all_metrics, step=self.current_epoch)
            
            # Check for improvement
            val_f1 = val_metrics.get("val_f1", 0)
            is_best = val_f1 > self.best_val_f1
            
            if is_best:
                self.best_val_f1 = val_f1
                self.best_val_loss = val_metrics.get("val_loss", float("inf"))
                no_improve = 0
            else:
                no_improve += 1
            
            # Save checkpoint
            if (self.current_epoch % save_every == 0) or is_best:
                self.save_checkpoint(
                    f"checkpoint_epoch_{self.current_epoch}.pt",
                    is_best=is_best
                )
            
            # Early stopping
            if no_improve >= patience:
                print(f"⚠️ Early stopping at epoch {self.current_epoch}")
                break
        
        print(f"\n✅ Training complete!")
        print(f"   Best val F1: {self.best_val_f1:.4f}")
        
        if self.use_wandb:
            wandb.finish()
        
        return self.best_val_f1
