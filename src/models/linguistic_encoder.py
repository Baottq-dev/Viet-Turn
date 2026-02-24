"""
Linguistic Encoder for MM-VAP-VI.

Combines PhoBERT (Vietnamese BERT) with VAPHuTuDetector (discourse marker detector)
to produce linguistic representations for turn-taking prediction.

Linguistically grounded in Vietnamese pragmatics:
- Tiểu từ tình thái (sentence-final modal particles) for turn-yielding
- Liên từ (conjunctions) for turn-holding
- Tín hiệu phản hồi (feedback signals) for backchannels
- N-gram matching for multi-word markers (thế à, đúng rồi, etc.)
- Position-sensitive classification (không = negation mid-sentence, question tag final)

Theoretical motivation — SFP-prosody complementarity hypothesis (Wakefield 2016):
  In tone languages like Vietnamese, F0 is occupied by lexical tone, so sentence-final
  particles take over interactional turn-management functions that intonation handles
  in non-tone languages. This creates a "dual-use F0 problem" (Levow 2005) that the
  HuTuDetector explicitly disambiguates.

Cross-linguistic precedent:
  - Japanese *ne* in 4 positions (Katagiri 2007, Pragmatics & Cognition)
  - Korean *ya* position-sensitive (Kim, Kim & Sohn 2021, Journal of Pragmatics Q1)
  - Mandarin SFPs as morphological turn-type markers (Levow 2005, SIGHAN/ACL)

References:
  - Wakefield (2016), SFPs and Intonation: Two Forms of the Same Thing
  - Ha & Grice (2017), Tone and Intonation in Discourse Management in Vietnamese
  - Sacks, Schegloff & Jefferson (1974), Turn-taking Systematics
  - Tran (2018), Teaching Final Particles in Vietnamese
  - Hoang (2025), Demonstratives as Utterance-Final Particles in Vietnamese
"""

import math
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, List, Set, Dict, Tuple


class VAPHuTuDetector(nn.Module):
    """
    Enhanced hư từ (discourse marker) detector for VAP.

    Key design choices:
    1. N-gram matching (unigram + bigram + trigram) for multi-word markers
    2. Position-sensitive type assignment: some markers change function
       at sentence-final position (không, rồi, mà, đi, thôi, chứ)
    3. Exponential position weighting: markers near end are more important
    4. Learned per-marker embeddings + per-type embeddings
    5. 4 categories: yield (0), hold (1), backchannel (2), turn_request (3), none (4)
    """

    # ── Marker Inventories ──────────────────────────────────────────
    # Sentence-final particles signaling turn completion / floor release
    YIELD_MARKERS: Set[str] = {
        # Confirmation-seeking (hỏi xác nhận)
        'nhé', 'nhỉ', 'hả', 'chứ', 'chưa', 'chăng', 'ư',
        # Politeness / softening (lịch sự)
        'ạ', 'nha', 'hen',
        # Directive completion (kết thúc yêu cầu)
        'đi', 'thôi', 'nào',
        # Assertive completion (khẳng định hoàn thành)
        'đấy', 'đây', 'cơ', 'kìa',
        # Completion aspect (hoàn thành) — position-sensitive
        'xong', 'hết',
    }

    # Markers that are YIELD only when sentence-final, else neutral/hold
    POSITION_SENSITIVE_YIELD: Set[str] = {
        'không',   # mid-sentence = negation, final = question tag
        'rồi',     # mid-sentence = "then"/aspect, final = completion
        'mà',      # mid-sentence = "but", final = insistence
        'à',       # mid-sentence = filler, final = question/surprise
        'chứ',     # can be hold ("chứ không phải") but usually yield at final
    }

    # Conjunctions / connectives signaling speaker continuation
    HOLD_MARKERS: Set[str] = {
        # Conjunctions (liên từ)
        'nhưng', 'vì', 'nên', 'nếu', 'khi', 'do',
        'còn', 'hay', 'với', 'cũng',
        # Topic / copula markers (thường hold)
        'thì', 'là',
        # "mà" default = hold (overridden to yield when sentence-final)
        'mà',
    }

    # Multi-word hold markers (liên từ đa âm tiết)
    HOLD_MARKERS_MULTI: Set[str] = {
        'tức là', 'nghĩa là', 'có nghĩa là',
        'ví dụ', 'ví dụ như',
        'thế nên', 'cho nên', 'vậy nên',
        'nói chung là', 'nói cách khác',
        'bởi vì', 'do đó', 'vì vậy',
    }

    # Feedback signals / acknowledgment tokens
    BACKCHANNEL_MARKERS: Set[str] = {
        # Simple acknowledgment (xác nhận đơn)
        'ừ', 'vâng', 'ờ', 'dạ', 'ừm', 'uhm',
        # Agreement
        'được',
        # Borrowed
        'ok', 'oke',
    }

    # Multi-word backchannel markers
    BACKCHANNEL_MARKERS_MULTI: Set[str] = {
        # Compound acknowledgment
        'ừ hử', 'à ha',
        # Surprise / reactive (Northern)
        'thế à', 'thế hả', 'thật à', 'thật hả',
        # Surprise / reactive (Southern)
        'vậy hả', 'vậy à', 'vậy sao',
        # Confirmatory
        'đúng rồi', 'phải rồi',
        # Continuation prompts
        'rồi sao', 'thế rồi sao',
        # Realization
        'ra vậy', 'ra thế',
    }

    # Turn-requesting markers (xin lượt nói)
    TURN_REQUEST_MARKERS: Set[str] = {
        'này', 'ơi',
    }

    TURN_REQUEST_MARKERS_MULTI: Set[str] = {
        'khoan đã', 'nhưng mà',
        'để tôi', 'để em',
        'cho tôi', 'cho em',
        'tôi nghĩ', 'em nghĩ',
    }

    # Maximum n-gram size to check
    MAX_NGRAM = 3

    def __init__(
        self,
        embedding_dim: int = 64,
        output_dim: int = 256,
        position_decay_alpha: float = 0.1,
    ):
        super().__init__()
        self.position_decay_alpha = position_decay_alpha

        # Build unified marker index (unigrams + multi-word)
        all_single = (
            self.YIELD_MARKERS | self.POSITION_SENSITIVE_YIELD |
            self.HOLD_MARKERS | self.BACKCHANNEL_MARKERS |
            self.TURN_REQUEST_MARKERS
        )
        all_multi = (
            self.HOLD_MARKERS_MULTI | self.BACKCHANNEL_MARKERS_MULTI |
            self.TURN_REQUEST_MARKERS_MULTI
        )
        self.all_markers = sorted(all_single | all_multi)
        self.marker_to_idx = {m: i for i, m in enumerate(self.all_markers)}
        self.num_markers = len(self.all_markers)

        # Pre-compute multi-word markers grouped by n-gram size for efficient lookup
        self._multi_by_n: Dict[int, Set[str]] = {}
        for m in all_multi:
            n = len(m.split())
            if n not in self._multi_by_n:
                self._multi_by_n[n] = set()
            self._multi_by_n[n].add(m)

        # Per-marker learned embeddings
        self.marker_embed = nn.Embedding(self.num_markers + 1, embedding_dim)  # +1 for no-marker

        # Marker type embeddings: yield=0, hold=1, backchannel=2, turn_request=3, none=4
        self.type_embed = nn.Embedding(5, embedding_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _get_marker_type(self, marker: str, is_near_end: bool) -> int:
        """
        Get marker type with position-sensitive classification.

        Args:
            marker: The marker string (may be multi-word).
            is_near_end: Whether the marker is within last 2 tokens of the utterance.

        Returns:
            Type index: 0=yield, 1=hold, 2=backchannel, 3=turn_request, 4=none
        """
        m = marker.lower()

        # Position-sensitive markers: only YIELD when near end
        if m in self.POSITION_SENSITIVE_YIELD:
            return 0 if is_near_end else 1  # yield if final, hold otherwise

        # Fixed categories (order matters: check multi-word first)
        if m in self.BACKCHANNEL_MARKERS_MULTI or m in self.BACKCHANNEL_MARKERS:
            return 2
        if m in self.TURN_REQUEST_MARKERS_MULTI or m in self.TURN_REQUEST_MARKERS:
            return 3
        if m in self.YIELD_MARKERS:
            return 0
        if m in self.HOLD_MARKERS_MULTI or m in self.HOLD_MARKERS:
            return 1
        return 4

    def _find_markers(self, tokens: List[str]) -> List[Tuple[str, int, int]]:
        """
        Find all markers in token list using n-gram matching.
        Longer matches take priority (greedy longest-match).

        Args:
            tokens: Lowercased token list.

        Returns:
            List of (marker_string, start_pos, end_pos) tuples.
            end_pos is exclusive.
        """
        num_tokens = len(tokens)
        found = []
        consumed = set()  # Track consumed token positions

        # Greedy longest-match: check trigrams, then bigrams, then unigrams
        for n in sorted(self._multi_by_n.keys(), reverse=True):
            for i in range(num_tokens - n + 1):
                if any(j in consumed for j in range(i, i + n)):
                    continue
                ngram = ' '.join(tokens[i:i + n])
                if ngram in self.marker_to_idx:
                    found.append((ngram, i, i + n))
                    for j in range(i, i + n):
                        consumed.add(j)

        # Then check unigrams not yet consumed
        for i in range(num_tokens):
            if i in consumed:
                continue
            token = tokens[i]
            if token in self.marker_to_idx:
                found.append((token, i, i + 1))
                consumed.add(i)

        # Sort by position
        found.sort(key=lambda x: x[1])
        return found

    def forward(self, text: str, device: torch.device = None) -> torch.Tensor:
        """
        Detect markers and return embedding.

        Args:
            text: Input text string.
            device: Target device.

        Returns:
            (output_dim,) marker representation vector.
        """
        if device is None:
            device = self.marker_embed.weight.device

        tokens = text.lower().split()
        num_tokens = len(tokens)

        if num_tokens == 0:
            no_marker = self.marker_embed(torch.tensor(self.num_markers, device=device))
            no_type = self.type_embed(torch.tensor(4, device=device))
            return self.output_proj(torch.cat([no_marker, no_type]))

        # Find all markers (n-gram matching)
        found_markers = self._find_markers(tokens)

        if not found_markers:
            no_marker = self.marker_embed(torch.tensor(self.num_markers, device=device))
            no_type = self.type_embed(torch.tensor(4, device=device))
            return self.output_proj(torch.cat([no_marker, no_type]))

        # Build embeddings with position-weighted aggregation
        marker_ids = []
        marker_types = []
        weights = []

        for marker_str, start_pos, end_pos in found_markers:
            marker_ids.append(self.marker_to_idx[marker_str])

            # Position-sensitive: is this marker near the end of the utterance?
            is_near_end = (end_pos >= num_tokens - 1)  # last or second-to-last token

            marker_types.append(self._get_marker_type(marker_str, is_near_end))

            # Exponential position weighting: closer to end = higher weight
            # Use the end position of the marker for weighting
            distance_to_end = num_tokens - end_pos
            weight = math.exp(-self.position_decay_alpha * distance_to_end)
            weights.append(weight)

        # Weighted mean of marker embeddings
        ids_tensor = torch.tensor(marker_ids, device=device)
        types_tensor = torch.tensor(marker_types, device=device)
        weights_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
        weights_tensor = weights_tensor / weights_tensor.sum()  # Normalize

        marker_embeds = self.marker_embed(ids_tensor)  # (K, embedding_dim)
        type_embeds = self.type_embed(types_tensor)      # (K, embedding_dim)

        weighted_marker = (marker_embeds * weights_tensor.unsqueeze(-1)).sum(dim=0)
        weighted_type = (type_embeds * weights_tensor.unsqueeze(-1)).sum(dim=0)

        combined = torch.cat([weighted_marker, weighted_type])
        return self.output_proj(combined)

    def forward_batch(self, texts: List[str], device: torch.device = None) -> torch.Tensor:
        """
        Batch forward for multiple texts.

        Args:
            texts: List of text strings.
            device: Target device.

        Returns:
            (B, output_dim) marker representations.
        """
        return torch.stack([self.forward(t, device) for t in texts])


class LinguisticEncoder(nn.Module):
    """
    Combined PhoBERT + HuTuDetector linguistic encoder.

    Input: text string(s)
    Output: (B, output_dim) linguistic representation
    """

    def __init__(
        self,
        pretrained: str = "vinai/phobert-base-v2",
        output_dim: int = 256,
        max_length: int = 256,
        freeze_embeddings: bool = True,
        freeze_layers: int = 8,
        use_hutu: bool = True,
        hutu_embedding_dim: int = 64,
        hutu_output_dim: int = 256,
        position_decay_alpha: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.max_length = max_length
        self.use_hutu = use_hutu

        # PhoBERT
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.phobert = AutoModel.from_pretrained(pretrained, use_safetensors=True)
        self.phobert_hidden = self.phobert.config.hidden_size  # 768

        # Freeze embeddings
        if freeze_embeddings:
            for param in self.phobert.embeddings.parameters():
                param.requires_grad = False

        # Freeze first N layers
        if freeze_layers > 0:
            for layer in self.phobert.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # PhoBERT projection
        self.phobert_proj = nn.Linear(self.phobert_hidden, output_dim)

        # HuTuDetector
        if use_hutu:
            self.hutu = VAPHuTuDetector(
                embedding_dim=hutu_embedding_dim,
                output_dim=hutu_output_dim,
                position_decay_alpha=position_decay_alpha,
            )
            # Combined projection: PhoBERT + HuTu -> output_dim
            self.combined_proj = nn.Sequential(
                nn.Linear(output_dim + hutu_output_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        texts: List[str],
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Encode a batch of texts.

        Args:
            texts: List of text strings (B,).
            device: Target device.

        Returns:
            (B, output_dim) linguistic representations.
        """
        if device is None:
            device = next(self.parameters()).device

        # Tokenize
        encoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # PhoBERT forward
        outputs = self.phobert(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )

        # [CLS] pooling
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        phobert_out = self.phobert_proj(cls_hidden)  # (B, output_dim)

        if self.use_hutu:
            # HuTuDetector
            hutu_out = self.hutu.forward_batch(texts, device)  # (B, hutu_output_dim)
            # Combine
            combined = torch.cat([phobert_out, hutu_out], dim=-1)
            output = self.combined_proj(combined)
        else:
            output = self.norm(phobert_out)

        return output

    def freeze_all(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n: int):
        """Unfreeze last N PhoBERT layers + projections."""
        # Unfreeze projections
        for param in self.phobert_proj.parameters():
            param.requires_grad = True
        if self.use_hutu:
            for param in self.hutu.parameters():
                param.requires_grad = True
            for param in self.combined_proj.parameters():
                param.requires_grad = True

        # Unfreeze last N PhoBERT layers
        total_layers = len(self.phobert.encoder.layer)
        for layer in self.phobert.encoder.layer[total_layers - n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
