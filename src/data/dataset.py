"""
PyTorch Dataset classes cho Viet-Turn training.
Hỗ trợ cả pre-cut segments và on-the-fly cutting.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


class TurnTakingDataset(Dataset):
    """
    Dataset cho turn-taking prediction.
    
    Mỗi sample = 1 audio segment + label
    """
    
    LABEL_TO_ID = {"YIELD": 0, "HOLD": 1, "BACKCHANNEL": 2}
    ID_TO_LABEL = {0: "YIELD", 1: "HOLD", 2: "BACKCHANNEL"}
    
    def __init__(
        self,
        data_path: str,
        audio_processor = None,
        use_precut: bool = True,
        audio_dir: Optional[str] = None,
        max_length_sec: float = 10.0,
        sample_rate: int = 16000
    ):
        """
        Args:
            data_path: Path to train.json / val.json / test.json
            audio_processor: AudioProcessor instance for feature extraction
            use_precut: If True, load pre-cut segment files. If False, cut on-the-fly
            audio_dir: Directory with original audio (needed if use_precut=False)
            max_length_sec: Maximum segment length in seconds
            sample_rate: Audio sample rate
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.segments = json.load(f)
        
        self.audio_processor = audio_processor
        self.use_precut = use_precut
        self.audio_dir = audio_dir
        self.max_length = int(max_length_sec * sample_rate)
        self.sample_rate = sample_rate
        
        # Filter valid segments
        self.segments = [s for s in self.segments if self._is_valid(s)]
        
        print(f"Loaded {len(self.segments)} segments from {data_path}")
    
    def _is_valid(self, segment: Dict) -> bool:
        """Check if segment is valid for training"""
        # Must have label
        if "label" not in segment:
            return False
        
        # Must have audio source
        if self.use_precut:
            return "audio_segment" in segment and Path(segment["audio_segment"]).exists()
        else:
            return "audio_file" in segment and "start" in segment and "end" in segment
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Returns:
            - features: Audio features tensor
            - label: Label ID (0, 1, or 2)
            - metadata: Dict with text, speaker, etc.
        """
        segment = self.segments[idx]
        
        # Load audio
        if self.use_precut:
            audio = self._load_precut(segment)
        else:
            audio = self._load_onthefly(segment)
        
        # Pad or truncate to max_length
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        elif len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        
        # Extract features
        if self.audio_processor is not None:
            features = self.audio_processor(audio)
        else:
            features = torch.FloatTensor(audio)
        
        # Get label
        label_str = segment.get("label", "YIELD").upper()
        label = self.LABEL_TO_ID.get(label_str, 0)
        
        # Metadata
        metadata = {
            "text": segment.get("text", ""),
            "speaker": segment.get("speaker", ""),
            "segment_id": segment.get("id", idx),
            "source_file": segment.get("source_file", "")
        }
        
        return features, label, metadata
    
    def _load_precut(self, segment: Dict) -> np.ndarray:
        """Load pre-cut audio segment"""
        audio_path = segment["audio_segment"]
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        return audio
    
    def _load_onthefly(self, segment: Dict) -> np.ndarray:
        """Load and cut audio on-the-fly"""
        audio_file = segment["audio_file"]
        audio_path = Path(self.audio_dir) / audio_file
        
        start = segment["start"]
        end = segment["end"]
        duration = end - start
        
        audio, _ = librosa.load(
            str(audio_path),
            sr=self.sample_rate,
            offset=start,
            duration=duration
        )
        
        return audio
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data"""
        counts = [0, 0, 0]
        
        for seg in self.segments:
            label_str = seg.get("label", "YIELD").upper()
            label_id = self.LABEL_TO_ID.get(label_str, 0)
            counts[label_id] += 1
        
        total = sum(counts)
        weights = [total / (3 * c) if c > 0 else 1.0 for c in counts]
        
        return torch.FloatTensor(weights)


def create_dataloader(
    data_path: str,
    audio_processor = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    use_precut: bool = True,
    audio_dir: Optional[str] = None
) -> DataLoader:
    """
    Create DataLoader for training/evaluation.
    
    Usage:
        from src.data.audio_processor import AudioProcessor
        from src.data.dataset import create_dataloader
        
        processor = AudioProcessor()
        train_loader = create_dataloader(
            "data/final/train.json",
            audio_processor=processor,
            batch_size=32
        )
        
        for features, labels, metadata in train_loader:
            # features: (B, 42, T)
            # labels: (B,)
            ...
    """
    dataset = TurnTakingDataset(
        data_path=data_path,
        audio_processor=audio_processor,
        use_precut=use_precut,
        audio_dir=audio_dir
    )
    
    # Custom collate to handle metadata
    def collate_fn(batch):
        features = torch.stack([b[0] for b in batch])
        labels = torch.LongTensor([b[1] for b in batch])
        metadata = [b[2] for b in batch]
        return features, labels, metadata
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
