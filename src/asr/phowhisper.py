"""
PhoWhisper ASR integration for streaming transcription.
Uses vinai/PhoWhisper for Vietnamese speech recognition.
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Dict, Optional, Generator
import numpy as np


class PhoWhisperASR:
    """
    PhoWhisper ASR wrapper for Vietnamese speech recognition.
    Supports streaming with sliding window approach.
    
    Usage:
        asr = PhoWhisperASR()
        text = asr.transcribe(audio_array)
        
        # Streaming
        for result in asr.stream_transcribe(audio_stream):
            print(result["text"])
    """
    
    def __init__(
        self,
        model_name: str = "vinai/PhoWhisper-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        language: str = "vi",
        sliding_window_sec: float = 3.0,
        stride_sec: float = 0.5
    ):
        self.device = device
        self.language = language
        self.sample_rate = 16000
        self.sliding_window = int(sliding_window_sec * self.sample_rate)
        self.stride = int(stride_sec * self.sample_rate)
        
        print(f"📦 Loading PhoWhisper: {model_name}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        print(f"   ✅ Loaded on {device}")
    
    @torch.no_grad()
    def transcribe(
        self,
        audio: np.ndarray,
        return_timestamps: bool = False
    ) -> Dict:
        """
        Transcribe audio array.
        
        Args:
            audio: Audio array, shape (samples,), sample rate 16kHz
            return_timestamps: If True, return word timestamps
        
        Returns:
            Dict with "text" and optionally "segments"
        """
        # Preprocess
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Generate
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language,
            task="transcribe"
        )
        
        generate_kwargs = {
            "forced_decoder_ids": forced_decoder_ids,
            "max_new_tokens": 448
        }
        
        if return_timestamps:
            generate_kwargs["return_timestamps"] = True
        
        predicted_ids = self.model.generate(inputs, **generate_kwargs)
        
        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        result = {"text": transcription.strip()}
        
        return result
    
    def stream_transcribe(
        self,
        audio_stream: Generator[np.ndarray, None, None]
    ) -> Generator[Dict, None, None]:
        """
        Streaming transcription with sliding window.
        
        Args:
            audio_stream: Generator yielding audio chunks
        
        Yields:
            Dict with "text", "start", "end" for each window
        """
        buffer = np.array([], dtype=np.float32)
        position = 0  # samples
        
        for chunk in audio_stream:
            buffer = np.concatenate([buffer, chunk])
            
            # Process when buffer is large enough
            while len(buffer) >= self.sliding_window:
                window = buffer[:self.sliding_window]
                
                result = self.transcribe(window)
                
                start_time = position / self.sample_rate
                end_time = (position + self.sliding_window) / self.sample_rate
                
                yield {
                    "text": result["text"],
                    "start": start_time,
                    "end": end_time
                }
                
                # Slide buffer
                buffer = buffer[self.stride:]
                position += self.stride
        
        # Process remaining buffer
        if len(buffer) > self.sample_rate * 0.5:  # At least 0.5 sec
            result = self.transcribe(buffer)
            
            start_time = position / self.sample_rate
            end_time = (position + len(buffer)) / self.sample_rate
            
            yield {
                "text": result["text"],
                "start": start_time,
                "end": end_time
            }
    
    def transcribe_file(self, audio_path: str) -> Dict:
        """Transcribe audio file."""
        import librosa
        
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return self.transcribe(audio)
