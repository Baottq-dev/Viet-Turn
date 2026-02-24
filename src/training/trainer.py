"""
VAPTrainer - 3-stage training for MM-VAP-VI.

Stage 1: Audio-only (freeze text + fusion)
Stage 2: Multimodal (unfreeze last N PhoBERT layers + fusion)
Stage 3: Full fine-tune (unfreeze all except CNN feature extractor)
"""

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional, List
import time
import json
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from src.models.model import MMVAPModel
from .losses import VAPLoss
from .augmentation import VAPAugmenter


class VAPTrainer:
    """
    3-stage trainer for MM-VAP-VI.

    Usage:
        model = MMVAPModel.from_config("configs/config.yaml")
        trainer = VAPTrainer(model, train_loader, val_loader, config_path)
        trainer.train()
    """

    def __init__(
        self,
        model: MMVAPModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config_path: str = "configs/config.yaml",
        device: str = "cuda",
        output_dir: str = "outputs/mm_vap",
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = self.config["training"]

        # Loss
        loss_cfg = train_cfg["loss"]
        self.criterion = VAPLoss(
            num_classes=self.config["data"]["num_classes"],
            transition_weight=loss_cfg.get("transition_weight", 1.0),
            transition_window_frames=int(loss_cfg.get("transition_window_ms", 500) / 20),
            use_focal=loss_cfg.get("type") == "focal",
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        )

        # Mixed precision
        self.use_amp = train_cfg.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Gradient accumulation
        self.accumulate_steps = train_cfg.get("accumulate_grad_batches", 1)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.patience = es_cfg.get("patience", 5)
        self.es_metric = es_cfg.get("metric", "val_loss")
        self.es_mode = es_cfg.get("mode", "min")

        # Checkpointing
        ckpt_cfg = train_cfg.get("checkpoint", {})
        self.save_every = ckpt_cfg.get("save_every_epoch", 5)
        self.keep_last = ckpt_cfg.get("keep_last", 3)

        # Augmentation
        self.augmenter = VAPAugmenter.from_config(self.config.get("augmentation"))

        # Tracking
        self.best_metric = float("inf") if self.es_mode == "min" else float("-inf")
        self.global_step = 0
        self.history = []

        # Resume state (set by load_checkpoint)
        self.resume_stage = None
        self.resume_epoch = None

        # Wandb
        log_cfg = self.config.get("logging", {}).get("wandb", {})
        self.use_wandb = HAS_WANDB and log_cfg.get("project") is not None
        self.log_every = log_cfg.get("log_every_steps", 50)
        if self.use_wandb:
            wandb.init(
                project=log_cfg["project"],
                entity=log_cfg.get("entity"),
                config=self.config,
            )

    def _create_optimizer(self, stage: int) -> AdamW:
        """Create optimizer with per-component learning rates for a stage."""
        param_groups = self.model.get_param_groups(stage, self.config["training"])
        weight_decay = self.config["training"].get("weight_decay", 0.01)

        return AdamW(param_groups, weight_decay=weight_decay)

    def _create_scheduler(self, optimizer: AdamW, num_epochs: int) -> torch.optim.lr_scheduler.LRScheduler:
        """Create OneCycleLR scheduler with cosine annealing and warmup."""
        total_steps = num_epochs * len(self.train_loader) // self.accumulate_steps
        warmup_steps = int(total_steps * self.config["training"].get("warmup_ratio", 0.05))

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in optimizer.param_groups],
            total_steps=total_steps,
            pct_start=warmup_steps / max(1, total_steps),
            anneal_strategy="cos",
        )
        return scheduler

    def train_epoch(
        self,
        optimizer: AdamW,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        stage: int,
        epoch: int,
    ) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_frames = 0
        num_batches = 0

        use_text = stage >= 2

        pbar = tqdm(
            self.train_loader,
            desc=f"Stage {stage} Epoch {epoch}",
        )

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Apply augmentation
            batch = self.augmenter(batch, use_text=use_text)

            audio = batch["audio_waveform"].to(self.device)
            labels = batch["vap_labels"].to(self.device)
            texts = batch["texts"]

            # Audio attention mask (for padded audio)
            audio_mask = None
            if "audio_lengths" in batch:
                max_len = audio.shape[1]
                lengths = batch["audio_lengths"]
                audio_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1).to(self.device)
                audio_mask = audio_mask.float()

            # Forward pass
            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(
                    audio_waveform=audio,
                    texts=texts,
                    audio_attention_mask=audio_mask,
                    use_text=use_text,
                )
                # Align frames: wav2vec2 CNN may produce fewer frames than labels
                T_logits = logits.shape[1]
                T_labels = labels.shape[1]
                if T_logits < T_labels:
                    labels = labels[:, :T_logits]
                elif T_logits > T_labels:
                    logits = logits[:, :T_labels, :]
                loss = self.criterion(logits, labels)
                loss = loss / self.accumulate_steps

            # Backward
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulate_steps == 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                self.global_step += 1

            # Tracking
            batch_loss = loss.item() * self.accumulate_steps
            total_loss += batch_loss
            num_batches += 1

            # Frame-level accuracy (on valid frames only)
            valid_mask = labels != -1
            if valid_mask.any():
                preds = logits.argmax(dim=-1)
                total_correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
                total_frames += valid_mask.sum().item()

            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "acc": f"{total_correct / max(1, total_frames):.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}",
            })

            # Wandb logging
            if self.use_wandb and self.global_step % self.log_every == 0:
                wandb.log({
                    "train/loss": batch_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/stage": stage,
                }, step=self.global_step)

        return {
            "train_loss": total_loss / max(1, num_batches),
            "train_acc": total_correct / max(1, total_frames),
        }

    @torch.no_grad()
    def validate(self, stage: int) -> Dict:
        """Validate on validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_frames = 0
        num_batches = 0

        use_text = stage >= 2

        for batch in self.val_loader:
            audio = batch["audio_waveform"].to(self.device)
            labels = batch["vap_labels"].to(self.device)
            texts = batch["texts"]

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
            # Align frames
            T_logits = logits.shape[1]
            T_labels = labels.shape[1]
            if T_logits < T_labels:
                labels = labels[:, :T_logits]
            elif T_logits > T_labels:
                logits = logits[:, :T_labels, :]
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            valid_mask = labels != -1
            if valid_mask.any():
                preds = logits.argmax(dim=-1)
                total_correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
                total_frames += valid_mask.sum().item()

        metrics = {
            "val_loss": total_loss / max(1, num_batches),
            "val_acc": total_correct / max(1, total_frames),
        }

        if self.use_wandb:
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=self.global_step)

        return metrics

    def _is_better(self, metric_value: float) -> bool:
        """Check if metric improved."""
        if self.es_mode == "min":
            return metric_value < self.best_metric
        return metric_value > self.best_metric

    def save_checkpoint(self, stage: int, epoch: int, is_best: bool = False,
                        optimizer=None, scheduler=None):
        """Save training checkpoint."""
        ckpt = {
            "stage": stage,
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
            "history": self.history,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()

        path = self.output_dir / f"checkpoint_s{stage}_e{epoch}.pt"
        torch.save(ckpt, path)

        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(ckpt, best_path)

        # Clean old checkpoints
        ckpts = sorted(self.output_dir.glob("checkpoint_s*.pt"), key=lambda p: p.stat().st_mtime)
        while len(ckpts) > self.keep_last:
            ckpts[0].unlink()
            ckpts.pop(0)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint and restore state for resuming.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Restore model weights
        self.model.load_state_dict(ckpt["model_state_dict"])

        # Restore trainer state
        self.global_step = ckpt.get("global_step", 0)
        self.best_metric = ckpt.get("best_metric", self.best_metric)
        self.history = ckpt.get("history", [])

        # Restore scaler state
        if self.use_amp and ckpt.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        # Set resume point
        self.resume_stage = ckpt.get("stage", 1)
        self.resume_epoch = ckpt.get("epoch", 0)

        # Store optimizer/scheduler states for later restoration in train_stage
        self._resume_optimizer_state = ckpt.get("optimizer_state_dict")
        self._resume_scheduler_state = ckpt.get("scheduler_state_dict")

    def train_stage(self, stage: int, start_epoch: int = 1):
        """
        Train a single stage.

        Args:
            stage: Stage number (1, 2, or 3).
            start_epoch: Epoch to start from (>1 when resuming mid-stage).
        """
        train_cfg = self.config["training"]
        stage_cfg = train_cfg[f"stage{stage}"]
        num_epochs = stage_cfg["epochs"]

        if num_epochs == 0:
            print(f"\n  Stage {stage}: skipped (epochs=0)")
            return

        print(f"\n{'='*60}")
        print(f"Stage {stage}: epochs {start_epoch}-{num_epochs}")
        params = self.model.count_parameters()
        print(f"  Trainable params: {params['total']['trainable']:,} / {params['total']['total']:,}")
        print(f"{'='*60}")

        optimizer = self._create_optimizer(stage)
        scheduler = self._create_scheduler(optimizer, num_epochs)

        # Restore optimizer/scheduler state if resuming within same stage
        if start_epoch > 1 and hasattr(self, "_resume_optimizer_state") and self._resume_optimizer_state is not None:
            optimizer.load_state_dict(self._resume_optimizer_state)
            self._resume_optimizer_state = None
        if start_epoch > 1 and hasattr(self, "_resume_scheduler_state") and self._resume_scheduler_state is not None:
            scheduler.load_state_dict(self._resume_scheduler_state)
            self._resume_scheduler_state = None

        no_improve = 0

        for epoch in range(start_epoch, num_epochs + 1):
            train_metrics = self.train_epoch(optimizer, scheduler, stage, epoch)
            val_metrics = self.validate(stage)

            all_metrics = {**train_metrics, **val_metrics, "stage": stage, "epoch": epoch}
            self.history.append(all_metrics)

            print(f"  Stage {stage} Epoch {epoch}: "
                  f"train_loss={train_metrics['train_loss']:.4f} "
                  f"train_acc={train_metrics['train_acc']:.3f}", end="")
            if val_metrics:
                print(f" val_loss={val_metrics['val_loss']:.4f} "
                      f"val_acc={val_metrics['val_acc']:.3f}", end="")
            print()

            # Check improvement
            metric_val = val_metrics.get(self.es_metric, train_metrics.get("train_loss"))
            is_best = self._is_better(metric_val)

            if is_best:
                self.best_metric = metric_val
                no_improve = 0
            else:
                no_improve += 1

            # Save checkpoint
            if epoch % self.save_every == 0 or is_best:
                self.save_checkpoint(stage, epoch, is_best, optimizer, scheduler)

            # Early stopping
            if no_improve >= self.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    def train(self):
        """Full 3-stage training, with resume support."""
        start_time = time.time()

        # Determine starting point
        start_stage = 1
        start_epoch = 1
        if self.resume_stage is not None:
            start_stage = self.resume_stage
            start_epoch = self.resume_epoch + 1  # resume from next epoch

            # If we finished the last epoch of the resumed stage, move to next stage
            stage_cfg = self.config["training"][f"stage{start_stage}"]
            if start_epoch > stage_cfg["epochs"]:
                start_stage += 1
                start_epoch = 1

            if start_stage > 3:
                print("All stages already completed in checkpoint.")
                return self.best_metric

            print(f"Resuming from stage {start_stage}, epoch {start_epoch}")

        for stage in [1, 2, 3]:
            if stage < start_stage:
                continue
            epoch_start = start_epoch if stage == start_stage else 1
            self.train_stage(stage, start_epoch=epoch_start)

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed/3600:.1f}h")
        print(f"Best {self.es_metric}: {self.best_metric:.4f}")

        # Save history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        if self.use_wandb:
            wandb.finish()

        return self.best_metric
