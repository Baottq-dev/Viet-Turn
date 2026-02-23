"""
Streaming inference for MM-VAP-VI.

Stateful inference module that processes audio in chunks,
maintains KV cache for the causal transformer, and integrates
text updates from streaming ASR.

Design:
- Audio buffer: accumulates raw audio samples
- Process every `chunk_ms` milliseconds (default: 200ms = 10 frames)
- KV cache: reuses past transformer attention keys/values
- Text cache: stores latest ASR transcript, updated asynchronously
- Output: per-frame VAP probabilities and turn-taking events
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from src.models.model import MMVAPModel
from src.models.transformer import KVCache
from src.utils.labels import vap_to_events, NUM_CLASSES, NUM_BINS


class StreamingMMVAP:
    """
    Stateful streaming inference wrapper for MM-VAP-VI.

    Uses KV cache in the causal transformer to avoid redundant computation
    on past frames. Only new audio frames are processed through the
    transformer, while past keys/values are reused from cache.

    Usage:
        model = MMVAPModel.from_config("config.yaml")
        model.load_state_dict(torch.load("best_model.pt")["model_state_dict"])

        streamer = StreamingMMVAP(model, device="cuda")

        # Process audio chunks as they arrive
        while audio_available:
            chunk = get_audio_chunk()  # (num_samples,) at 16kHz
            results = streamer.process_audio(chunk)
            if results:
                print(results["events"])

        # Optionally update text from ASR
        streamer.update_text("xin chào các bạn")
    """

    def __init__(
        self,
        model: MMVAPModel,
        device: str = "cuda",
        chunk_ms: int = 200,
        max_context_frames: int = 1000,
        sample_rate: int = 16000,
        frame_hz: int = 50,
        shift_threshold: float = 0.5,
        bc_threshold: float = 0.3,
        use_kv_cache: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        self.samples_per_frame = sample_rate // frame_hz  # 320
        self.chunk_samples = int(chunk_ms / 1000 * sample_rate)
        self.chunk_frames = int(chunk_ms / 1000 * frame_hz)
        self.max_context_frames = max_context_frames

        self.shift_threshold = shift_threshold
        self.bc_threshold = bc_threshold
        self.use_kv_cache = use_kv_cache

        # State
        self._audio_buffer = torch.tensor([], dtype=torch.float32)
        self._text_cache = ""
        self._total_frames_processed = 0
        self._probs_history: List[torch.Tensor] = []
        self._kv_cache: Optional[KVCache] = None

    def reset(self):
        """Reset all state for a new conversation."""
        self._audio_buffer = torch.tensor([], dtype=torch.float32)
        self._text_cache = ""
        self._total_frames_processed = 0
        self._probs_history = []
        self._kv_cache = None

    def update_text(self, text: str):
        """
        Update the current ASR transcript.

        Called asynchronously when streaming ASR produces new text.
        The text is used on the next audio processing call.

        Note: Text updates invalidate the KV cache since linguistic
        features change. Cache is automatically reset on next process_audio().
        """
        if text != self._text_cache:
            self._text_cache = text
            # Invalidate cache when text changes (linguistic features differ)
            self._kv_cache = None

    @torch.no_grad()
    def process_audio(
        self,
        audio_chunk: torch.Tensor,
        text: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Process a chunk of audio and return predictions.

        When KV cache is enabled, only new frames are processed through the
        transformer, reusing cached keys/values for past frames.

        Args:
            audio_chunk: (num_samples,) raw audio at 16kHz.
            text: Optional text override (uses cached text if None).

        Returns:
            Dict with predictions, or None if buffer not ready.
            {
                "probs": (num_new_frames, 256) frame probabilities,
                "p_now": (num_new_frames, 2) per-speaker p_now,
                "events": List of event dicts,
                "frame_offset": int, starting frame index,
            }
        """
        if text is not None and text != self._text_cache:
            self._text_cache = text
            self._kv_cache = None  # Invalidate cache on text change

        # Accumulate audio
        self._audio_buffer = torch.cat([self._audio_buffer, audio_chunk.cpu()])

        # Check if we have enough for processing
        buffer_samples = self._audio_buffer.shape[0]
        buffer_frames = buffer_samples // self.samples_per_frame

        if buffer_frames < self.chunk_frames:
            return None

        # Determine how much audio to process
        context_samples = min(
            buffer_samples,
            self.max_context_frames * self.samples_per_frame,
        )

        # Check if context was trimmed (cache must be invalidated)
        if buffer_samples > context_samples and self._kv_cache is not None:
            self._kv_cache = None

        # Trim audio to context window
        audio_input = self._audio_buffer[-context_samples:].to(self.device).unsqueeze(0)

        # Forward pass
        use_text = len(self._text_cache) > 0

        # Acoustic encoder + fusion (run on full context — these are fast)
        acoustic = self.model.acoustic_encoder(audio_input)
        if use_text:
            linguistic = self.model.linguistic_encoder([self._text_cache])
            fused = self.model.fusion(acoustic, linguistic)
        else:
            fused = acoustic

        # Transformer with optional KV cache
        if self.use_kv_cache and self._kv_cache is not None:
            # Incremental: only pass new frames through transformer
            new_frames_input = fused[:, -self.chunk_frames:]
            contextualized, self._kv_cache = self.model.transformer(
                new_frames_input, kv_cache=self._kv_cache,
            )
        else:
            # Full forward (first chunk or cache invalidated)
            contextualized, self._kv_cache = self.model.transformer(fused)

        # Projection head
        logits = self.model.projection_head(contextualized)
        probs = F.softmax(logits[0], dim=-1)  # (T, 256)

        # Only return predictions for new frames
        new_frames = min(self.chunk_frames, probs.shape[0])
        new_probs = probs[-new_frames:]

        # Compute p_now for each speaker
        class_indices = torch.arange(NUM_CLASSES, device=probs.device)
        s0_mask = ((class_indices >> 0) & 1).bool()
        s1_mask = ((class_indices >> NUM_BINS) & 1).bool()
        p_now_0 = new_probs[:, s0_mask].sum(dim=-1)
        p_now_1 = new_probs[:, s1_mask].sum(dim=-1)
        p_now = torch.stack([p_now_0, p_now_1], dim=-1)

        # Detect events from new frames
        events = vap_to_events(
            new_probs,
            shift_threshold=self.shift_threshold,
            bc_threshold=self.bc_threshold,
        )

        # Adjust event frame indices to global offset
        frame_offset = self._total_frames_processed
        for e in events:
            e["start_frame"] += frame_offset
            e["end_frame"] += frame_offset

        # Update state
        self._total_frames_processed += new_frames
        self._probs_history.append(new_probs.cpu())

        # Trim audio buffer to keep only context
        keep_samples = self.max_context_frames * self.samples_per_frame
        if self._audio_buffer.shape[0] > keep_samples:
            self._audio_buffer = self._audio_buffer[-keep_samples:]

        return {
            "probs": new_probs.cpu(),
            "p_now": p_now.cpu(),
            "events": events,
            "frame_offset": frame_offset,
            "num_frames": new_frames,
        }

    def get_current_state(self) -> Dict:
        """Get current streaming state summary."""
        return {
            "total_frames": self._total_frames_processed,
            "total_seconds": self._total_frames_processed / self.frame_hz,
            "buffer_samples": self._audio_buffer.shape[0],
            "current_text": self._text_cache[:100] + "..." if len(self._text_cache) > 100 else self._text_cache,
            "kv_cache_active": self._kv_cache is not None,
        }
