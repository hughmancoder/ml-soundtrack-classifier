import glob
import os
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from soundtrack_classifier.ingestion.degradation import SyntheticDegradation


def slice_audio_file(input_path: str, output_dir: str, duration_sec: float = 3.0, sr: int = 44100, rms_threshold: float = 0.01):
    """
    Slices a clean audio recording into exactly `duration_sec` chunks.
    Discards silent or near-silent chunks using RMS energy thresholding.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio, _ = librosa.load(input_path, sr=sr, mono=True)

    chunk_length_samples = int(duration_sec * sr)
    num_chunks = len(audio) // chunk_length_samples

    base_name = Path(input_path).stem
    saved_count = 0

    for i in range(num_chunks):
        start_idx = i * chunk_length_samples
        end_idx = start_idx + chunk_length_samples
        chunk = audio[start_idx:end_idx]

        # Calculate RMS energy to discard silence
        rms = librosa.feature.rms(y=chunk).mean()
        if rms > rms_threshold:
            out_file = os.path.join(output_dir, f"{base_name}_chunk{i:03d}.wav")
            sf.write(out_file, chunk, sr)
            saved_count += 1

    print(f"Sliced {input_path} into {saved_count} valid {duration_sec}s chunks.")
    return saved_count


class SoundtrackCNNDataset(Dataset):
    """
    CNN Dataset: Loads `duration_sec` .wav files.
    Outputs raw audio tensors so that CQT can be computed dynamically on batches in the training loop.
    Supports on-the-fly synthetic degradation.
    """
    def __init__(self, data_dir: str, class_to_idx: dict, sr: int = 44100, duration_sec: float = 3.0, degradation_config: dict = None):
        self.data_dir = data_dir
        self.class_to_idx = class_to_idx
        self.sr = sr
        self.num_samples = int(duration_sec * sr)

        self.degradation_config = degradation_config or {}
        self.degradation_enabled = self.degradation_config.get('enabled', False)
        self.degradation_prob = float(self.degradation_config.get('probability', 0.5))
        if self.degradation_prob > 1.0:
            self.degradation_prob /= 100.0

        if self.degradation_enabled:
            self.degrader = None # Initialize lazily in __getitem__ to prevent multiprocessing pickle errors

        # Gather all .wav files in the subdirectories according to the provided classes
        self.filepaths = []
        self.labels = []

        for class_name, idx in class_to_idx.items():
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for f in glob.glob(os.path.join(class_dir, "*.wav")):
                    self.filepaths.append(f)
                    self.labels.append(idx)

        print(f"Loaded dataset from {data_dir} with {len(self.filepaths)} total files.")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]

        # Load raw audio (librosa is fast enough for small 3s files, but soundfile is faster)
        audio, _ = librosa.load(path, sr=self.sr, mono=True)

        # Pad or randomly crop to exact length to ensure consistent batching
        if len(audio) < self.num_samples:
            audio = np.pad(audio, (0, self.num_samples - len(audio)))
        else:
            max_start = len(audio) - self.num_samples
            start_idx = random.randint(0, max_start)
            audio = audio[start_idx:start_idx + self.num_samples]

        # On-the-fly Data Augmentation: Synthetic Degradation
        if self.degradation_enabled and random.random() < self.degradation_prob:
            if self.degrader is None:
                self.degrader = SyntheticDegradation(sr=self.sr)
            audio = self.degrader.apply(audio)

        # Convert to tensor and return RAW audio
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        return audio_tensor, label
