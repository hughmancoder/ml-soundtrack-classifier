import os
import glob
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset

from soundtrack_classifier.ingestion.degradation import SyntheticDegradation


class ArchivalSoundtrackRNNDataset(Dataset):
    """
    Dataset loader for the Archival Instrument CRNN.
    
    Key Design Principles:
    - 6-to-8 Second Operational Window:
        Dynamically extracts random 6-to-8s crops (default 8.0s) from continuous scene recordings 
        or pre-segmented clips. This captures complete expressive gestures (attacks, sustained 
        vibrato, slides, decays) and provides sufficient temporal support for stable CQT filters.
    - Noise Contrast Learning:
        Exposes the recurrent model to longer temporal baselines to distinguish steady-state 
        optical noise from transient musical phrasing.
    - On-The-Fly Degradation:
        Applies synthetic 1960s optical film degradation (surface hiss, crackle, bandlimiting) 
        during training.
    """
    def __init__(
        self,
        data_dir: str,
        class_to_idx: Dict[str, int],
        sr: int = 22050,
        crop_duration_sec: float = 8.0,
        random_crop: bool = True,
        degradation_config: Optional[dict] = None,
        rms_threshold: float = 0.005
    ):
        self.data_dir = data_dir
        self.class_to_idx = class_to_idx
        self.sr = sr
        self.crop_duration_sec = crop_duration_sec
        self.num_samples = int(crop_duration_sec * sr)
        self.random_crop = random_crop
        self.rms_threshold = rms_threshold
        
        self.degradation_config = degradation_config or {}
        self.degradation_enabled = self.degradation_config.get("enabled", False)
        self.degradation_prob = float(self.degradation_config.get("probability", 0.5))
        if self.degradation_prob > 1.0: # Handle percentage format (e.g. 50 or 80)
            self.degradation_prob /= 100.0

        self.degrader: Optional[SyntheticDegradation] = None # Lazy init to prevent multiprocessing issues

        # Scan for audio files in class subdirectories
        self.filepaths: List[str] = []
        self.labels: List[int] = []

        for class_name, idx in class_to_idx.items():
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                files = (
                    glob.glob(os.path.join(class_dir, "*.wav")) + 
                    glob.glob(os.path.join(class_dir, "*.mp3")) +
                    glob.glob(os.path.join(class_dir, "*.flac"))
                )
                for f in files:
                    self.filepaths.append(f)
                    self.labels.append(idx)

        print(f"[ArchivalSoundtrackRNNDataset] Discovered {len(self.filepaths)} audio files across {len(class_to_idx)} classes in {data_dir}.")

    def __len__(self) -> int:
        return len(self.filepaths)

    def _pad_audio(self, audio: np.ndarray) -> np.ndarray:
        """Pads short audio cleanly by tiling/reflecting to avoid abrupt silence edges."""
        if len(audio) >= self.num_samples:
            return audio
            
        repeat_factor = int(np.ceil(self.num_samples / max(1, len(audio))))
        tiled = np.tile(audio, repeat_factor)
        return tiled[:self.num_samples]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.filepaths[idx]
        label = self.labels[idx]

        try:
            # Read audio header first to determine length if possible
            info = sf.info(path)
            file_sr = info.samplerate
            file_frames = info.frames
            target_file_frames = int(self.crop_duration_sec * file_sr)

            if file_frames > target_file_frames:
                if self.random_crop:
                    max_start = file_frames - target_file_frames
                    start_frame = random.randint(0, max_start)
                else:
                    start_frame = max(0, (file_frames - target_file_frames) // 2)
                
                audio, _ = sf.read(path, start=start_frame, stop=start_frame + target_file_frames, dtype="float32", always_2d=False)
            else:
                audio, _ = sf.read(path, dtype="float32", always_2d=False)

            # Ensure mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Resample if needed
            if file_sr != self.sr:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=self.sr)

        except Exception:
            # Fallback to librosa if soundfile encounters an edge case
            audio, _ = librosa.load(path, sr=self.sr, mono=True)

        # Pad if shorter than target samples
        if len(audio) < self.num_samples:
            audio = self._pad_audio(audio)
        elif len(audio) > self.num_samples:
            if self.random_crop:
                start_idx = random.randint(0, len(audio) - self.num_samples)
            else:
                start_idx = (len(audio) - self.num_samples) // 2
            audio = audio[start_idx:start_idx + self.num_samples]

        # Apply synthetic degradation during training
        if self.degradation_enabled and random.random() < self.degradation_prob:
            if self.degrader is None:
                self.degrader = SyntheticDegradation(sr=self.sr)
            try:
                audio = self.degrader.apply(audio)
            except Exception:
                pass

        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        return audio_tensor, label
