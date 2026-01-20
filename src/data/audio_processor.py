"""
Audio feature extraction for Vietnamese turn-taking prediction.
Features: Log-Mel Spectrogram + F0 Pitch + Energy
"""

import torch
import librosa
import numpy as np
from typing import Tuple, Optional

try:
    import parselmouth
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False
    print("Warning: parselmouth not installed. F0 extraction will use librosa fallback.")


class AudioProcessor:
    """Extract acoustic features optimized for Vietnamese prosody."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_ms: int = 20,
        frame_shift_ms: int = 10,
        n_mels: int = 40,
        n_fft: int = 512,
        f_min: float = 50.0,
        f_max: float = 400.0,  # Vietnamese F0 range
    ):
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * frame_length_ms / 1000)
        self.frame_shift = int(sample_rate * frame_shift_ms / 1000)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        
    def extract_mel(self, audio: np.ndarray) -> np.ndarray:
        """Extract log-mel spectrogram."""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.frame_shift,
            win_length=self.frame_length,
            n_mels=self.n_mels,
            fmin=80,
            fmax=self.sample_rate // 2
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel  # Shape: (n_mels, time)
    
    def extract_f0(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 contour using Praat/Parselmouth or librosa fallback."""
        if HAS_PARSELMOUTH:
            snd = parselmouth.Sound(audio, self.sample_rate)
            pitch = snd.to_pitch(
                time_step=self.frame_shift / self.sample_rate,
                pitch_floor=self.f_min,
                pitch_ceiling=self.f_max
            )
            f0 = pitch.selected_array['frequency']
            f0[f0 == 0] = np.nan
        else:
            # Fallback to librosa pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=self.f_min,
                fmax=self.f_max,
                sr=self.sample_rate,
                hop_length=self.frame_shift
            )
        
        f0 = np.nan_to_num(f0, nan=0.0)
        return f0  # Shape: (time,)
    
    def extract_energy(self, audio: np.ndarray) -> np.ndarray:
        """Extract frame-level energy (RMS)."""
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.frame_shift
        )[0]
        return energy  # Shape: (time,)
    
    def normalize(self, features: np.ndarray, axis: int = -1) -> np.ndarray:
        """Z-score normalization."""
        mean = features.mean(axis=axis, keepdims=True)
        std = features.std(axis=axis, keepdims=True) + 1e-8
        return (features - mean) / std
    
    def __call__(self, audio: np.ndarray) -> torch.Tensor:
        """Extract all features and concatenate."""
        mel = self.extract_mel(audio)        # (40, T)
        f0 = self.extract_f0(audio)          # (T,)
        energy = self.extract_energy(audio)  # (T,)
        
        # Align lengths
        min_len = min(mel.shape[1], len(f0), len(energy))
        mel = mel[:, :min_len]
        f0 = f0[:min_len]
        energy = energy[:min_len]
        
        # Normalize
        mel = self.normalize(mel, axis=1)
        f0 = self.normalize(f0)
        energy = self.normalize(energy)
        
        # Concatenate: (42, T) = 40 mel + 1 f0 + 1 energy
        features = np.vstack([
            mel,
            f0.reshape(1, -1),
            energy.reshape(1, -1)
        ])
        
        return torch.FloatTensor(features)
    
    def process_file(self, audio_path: str) -> torch.Tensor:
        """Process audio file and return features."""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return self(audio)
