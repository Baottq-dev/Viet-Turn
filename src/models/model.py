"""
MM-VAP-VI: Multimodal Voice Activity Projection for Vietnamese.

Full model that combines acoustic encoder, linguistic encoder,
cross-attention fusion, causal transformer, and VAP projection head.
"""

import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional

from .acoustic_encoder import AcousticEncoder
from .linguistic_encoder import LinguisticEncoder
from .fusion import CrossAttentionFusion, build_fusion
from .transformer import CausalTransformer
from .projection_head import VAPProjectionHead


class MMVAPModel(nn.Module):
    """
    Full MM-VAP-VI model.

    Pipeline:
        audio_waveform → AcousticEncoder → (B, T, 256)
        texts → LinguisticEncoder → (B, 256)
        (acoustic, linguistic) → CrossAttentionFusion → (B, T, 256)
        fused → CausalTransformer → (B, T, 256)
        contextualized → VAPProjectionHead → (B, T, 256_classes)
    """

    def __init__(
        self,
        acoustic_encoder: AcousticEncoder,
        linguistic_encoder: LinguisticEncoder,
        fusion: CrossAttentionFusion,
        transformer: CausalTransformer,
        projection_head: VAPProjectionHead,
    ):
        super().__init__()
        self.acoustic_encoder = acoustic_encoder
        self.linguistic_encoder = linguistic_encoder
        self.fusion = fusion
        self.transformer = transformer
        self.projection_head = projection_head

    def forward(
        self,
        audio_waveform: torch.Tensor,
        texts: List[str],
        audio_attention_mask: Optional[torch.Tensor] = None,
        use_text: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            audio_waveform: (B, num_samples) raw audio at 16kHz.
            texts: List of B text strings.
            audio_attention_mask: (B, num_samples) mask for padded audio.
            use_text: If False, skip linguistic branch (for audio-only training).

        Returns:
            logits: (B, T, 256) VAP class logits.
        """
        # Acoustic encoding
        acoustic = self.acoustic_encoder(
            audio_waveform,
            attention_mask=audio_attention_mask,
        )  # (B, T, dim)

        if use_text:
            # Linguistic encoding
            linguistic = self.linguistic_encoder(texts)  # (B, dim)

            # Cross-attention fusion
            fused = self.fusion(acoustic, linguistic)  # (B, T, dim)
        else:
            # Audio-only mode: skip fusion
            fused = acoustic

        # Causal transformer
        contextualized, _ = self.transformer(fused)  # (B, T, dim)

        # VAP projection
        logits = self.projection_head(contextualized)  # (B, T, 256)

        return logits

    @classmethod
    def from_config(cls, config_path: str) -> "MMVAPModel":
        """Build model from YAML config file."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        ac_cfg = cfg["acoustic_encoder"]
        acoustic_encoder = AcousticEncoder(
            pretrained=ac_cfg["pretrained"],
            encoder_type=ac_cfg.get("type", "wav2vec2"),
            output_dim=ac_cfg["output_dim"],
            freeze_layers=ac_cfg.get("freeze_layers", 8),
            gradient_checkpointing=ac_cfg.get("gradient_checkpointing", True),
        )

        lc_cfg = cfg["linguistic_encoder"]
        pb_cfg = lc_cfg["phobert"]
        ht_cfg = lc_cfg.get("hutu_detector", {})
        linguistic_encoder = LinguisticEncoder(
            pretrained=pb_cfg["pretrained"],
            output_dim=lc_cfg["combined_dim"],
            max_length=pb_cfg.get("max_length", 256),
            freeze_embeddings=pb_cfg.get("freeze_embeddings", True),
            freeze_layers=pb_cfg.get("freeze_layers", 8),
            use_hutu=ht_cfg.get("enabled", True),
            hutu_embedding_dim=ht_cfg.get("embedding_dim", 64),
            hutu_output_dim=ht_cfg.get("output_dim", 256),
            position_decay_alpha=ht_cfg.get("position_decay_alpha", 0.1),
        )

        f_cfg = cfg["fusion"]
        fusion_kwargs = {
            "dim": f_cfg["dim"],
            "num_heads": f_cfg.get("num_heads", 4),
            "dropout": f_cfg.get("dropout", 0.1),
        }
        if "num_latents" in f_cfg:
            fusion_kwargs["num_latents"] = f_cfg["num_latents"]
        fusion = build_fusion(
            fusion_type=f_cfg.get("type", "cross_attention"),
            **fusion_kwargs,
        )

        t_cfg = cfg["transformer"]
        transformer = CausalTransformer(
            num_layers=t_cfg["num_layers"],
            dim=t_cfg["dim"],
            num_heads=t_cfg["num_heads"],
            ffn_dim=t_cfg["ffn_dim"],
            dropout=t_cfg.get("dropout", 0.1),
            use_alibi=t_cfg.get("use_alibi", True),
            use_gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        )

        p_cfg = cfg["projection_head"]
        projection_head = VAPProjectionHead(
            input_dim=t_cfg["dim"],
            hidden_dim=p_cfg["hidden_dim"],
            num_classes=p_cfg["num_classes"],
        )

        return cls(
            acoustic_encoder=acoustic_encoder,
            linguistic_encoder=linguistic_encoder,
            fusion=fusion,
            transformer=transformer,
            projection_head=projection_head,
        )

    def get_param_groups(self, stage: int, config: Dict) -> List[Dict]:
        """
        Get parameter groups with per-component learning rates for a training stage.

        Args:
            stage: Training stage (1, 2, or 3).
            config: Training config dict with stage-specific LR settings.

        Returns:
            List of param group dicts for optimizer.
        """
        stage_cfg = config[f"stage{stage}"]
        lr_cfg = stage_cfg["lr"]

        # Apply freeze/unfreeze based on stage
        if stage == 1:
            self.linguistic_encoder.freeze_all()
            for param in self.fusion.parameters():
                param.requires_grad = False
        elif stage == 2:
            self.linguistic_encoder.unfreeze_last_n_layers(
                stage_cfg.get("unfreeze_phobert_layers", 2)
            )
            for param in self.fusion.parameters():
                param.requires_grad = True
        elif stage == 3:
            self.acoustic_encoder.unfreeze_all()
            self.linguistic_encoder.unfreeze_all()
            for param in self.fusion.parameters():
                param.requires_grad = True

        param_groups = []
        components = {
            "acoustic_encoder": self.acoustic_encoder,
            "linguistic_encoder": self.linguistic_encoder,
            "fusion": self.fusion,
            "transformer": self.transformer,
            "projection_head": self.projection_head,
        }

        for name, module in components.items():
            trainable = [p for p in module.parameters() if p.requires_grad]
            if trainable and name in lr_cfg:
                param_groups.append({
                    "params": trainable,
                    "lr": lr_cfg[name],
                    "name": name,
                })

        return param_groups

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per component."""
        components = {
            "acoustic_encoder": self.acoustic_encoder,
            "linguistic_encoder": self.linguistic_encoder,
            "fusion": self.fusion,
            "transformer": self.transformer,
            "projection_head": self.projection_head,
        }

        counts = {}
        total = 0
        trainable = 0
        for name, module in components.items():
            n_params = sum(p.numel() for p in module.parameters())
            n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {"total": n_params, "trainable": n_trainable}
            total += n_params
            trainable += n_trainable

        counts["total"] = {"total": total, "trainable": trainable}
        return counts
