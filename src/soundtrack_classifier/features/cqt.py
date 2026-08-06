import torch
import numpy as np
import librosa
from nnAudio.features.cqt import CQT1992v2

class CQTExtractor:
    """
    Extracts Constant-Q Transform (CQT) spectrograms from audio stems.
    Uses nnAudio for fast, GPU-accelerated extraction.
    """
    def __init__(self, sr: int = 44100, fmin: float = 32.7, n_bins: int = 84, bins_per_octave: int = 12, device: str = 'cpu'):
        self.sr = sr
        self.device = device
        
        # Initialize the CQT layer
        self.cqt_layer = CQT1992v2(
            sr=sr,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave
        ).to(self.device)

    def extract_from_file(self, audio_path: str) -> torch.Tensor:
        """
        Loads an audio file and computes its CQT spectrogram.
        Returns:
            A PyTorch tensor of shape (1, n_bins, time_steps).
        """
        # Load audio using librosa (nnAudio expects tensor, librosa returns numpy array)
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        
        # Convert to tensor and add batch dimension
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Extract CQT
        cqt_spectrogram = self.cqt_layer(audio_tensor)
        
        return cqt_spectrogram

    def save_features(self, cqt_tensor: torch.Tensor, output_path: str):
        """
        Saves the CQT tensor to disk as a .pt file for later use in Transformer training.
        """
        torch.save(cqt_tensor.cpu(), output_path)
        print(f"Saved CQT features to {output_path}")

    def load_features(self, input_path: str) -> torch.Tensor:
        """
        Loads saved CQT features from disk.
        """
        return torch.load(input_path).to(self.device)
