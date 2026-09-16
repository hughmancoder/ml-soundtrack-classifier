import os
import glob
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import librosa
import torch
import yaml
from torch.utils.data import Dataset

from soundtrack_classifier.ingestion.degradation import SyntheticDegradation


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Loads configuration from yaml file."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


class ArchivalSoundtrackRNNDataset(Dataset):
    """
    Dataset loader for the Archival Instrument CRNN.
    
    Key Design Principles:
    - 6-to-8 Second Operational Window:
        Dynamically extracts random 6-to-8s crops from continuous scene recordings 
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
        data_dir: Optional[str] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        sr: Optional[int] = None,
        crop_duration_sec: Optional[float] = None,
        random_crop: bool = True,
        degradation_config: Optional[dict] = None,
        rms_threshold: Optional[float] = None,
        features_dir: Optional[str] = None,
        precompute_features: Optional[bool] = None,
        config: Optional[dict] = None
    ):
        if config is None:
            config = load_config()

        crnn_cfg = config.get("crnn", {})
        crnn_audio = crnn_cfg.get("audio", {})
        crnn_inf = crnn_cfg.get("inference", {})
        crnn_train = crnn_cfg.get("training", {})

        self.data_dir = data_dir or config.get("paths", {}).get("data_dir", "data/train")
        self.sr = sr if sr is not None else int(crnn_audio.get("sample_rate", 22050))
        self.crop_duration_sec = crop_duration_sec if crop_duration_sec is not None else float(crnn_audio.get("crop_duration_sec", 8.0))
        self.num_samples = int(self.crop_duration_sec * self.sr)
        self.random_crop = random_crop
        self.rms_threshold = rms_threshold if rms_threshold is not None else float(crnn_inf.get("rms_threshold", 0.005))
        self.cqt_hop = int(crnn_audio.get("cqt", {}).get("hop_length", 512))

        # Precompute features handling:
        # Defaults to True in config.yaml to save CQT computation time.
        # If False, audio is loaded and transformed dynamically on the fly.
        if precompute_features is None:
            cfg_precompute = crnn_train.get("precompute_features")
            if cfg_precompute is not None:
                self.precompute_features = bool(cfg_precompute)
            else:
                # Default to False only if custom data_dir passed without config (e.g. unit tests)
                self.precompute_features = False if (data_dir and data_dir != "data/train" and not features_dir) else True
        else:
            self.precompute_features = bool(precompute_features)

        feat_candidate = features_dir or crnn_train.get("features_dir", "data/features_cqt_hpss")
        self.features_dir = feat_candidate
        self.features_mode = False

        crnn_degradation = crnn_cfg.get("degradation", {})
        self.degradation_config = degradation_config or crnn_degradation or config.get("degradation", {})
        self.degradation_enabled = self.degradation_config.get("enabled", False)
        self.degradation_prob = float(self.degradation_config.get("probability", 0.8))
        if self.degradation_prob > 1.0: # Handle percentage format (e.g. 50 or 80)
            self.degradation_prob /= 100.0

        self.degrader: Optional[SyntheticDegradation] = None # Lazy init to prevent multiprocessing issues

        if class_to_idx is None:
            train_labels = config.get("train_labels", [
                "sheng", "dizi", "erhu", "pipa", "suona", "guzheng",
                "yangqin", "percussion", "strings", "woodwind", "brass", "nonmusic"
            ])
            class_to_idx = {c: i for i, c in enumerate(train_labels)}
        self.class_to_idx = class_to_idx

        self.filepaths: List[str] = []
        self.labels: List[int] = []

        # 1. Precomputed features mode
        if self.precompute_features:
            cache_has_files = False
            if os.path.isdir(self.features_dir):
                for class_name in self.class_to_idx.keys():
                    class_feat_dir = os.path.join(self.features_dir, class_name)
                    if os.path.isdir(class_feat_dir) and glob.glob(os.path.join(class_feat_dir, "*.pt")):
                        cache_has_files = True
                        break

            # If cache is missing or empty, automatically run offline precomputation
            if not cache_has_files:
                print(f"[*] precompute_features=True but cache '{self.features_dir}' is missing or empty.")
                print(f"[*] Running precompute_features to cache 2-channel CQT+HPSS tensors...")
                import sys
                from pathlib import Path
                root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
                if root_dir not in sys.path:
                    sys.path.insert(0, root_dir)
                from scripts.precompute_cqt_hpss import precompute_features as run_precompute
                run_precompute(data_dir=self.data_dir, out_dir=self.features_dir, sr=self.sr)

            for class_name, idx in self.class_to_idx.items():
                class_feat_dir = os.path.join(self.features_dir, class_name)
                if os.path.isdir(class_feat_dir):
                    f_files = sorted(glob.glob(os.path.join(class_feat_dir, "*.pt")))
                    for f in f_files:
                        self.filepaths.append(f)
                        self.labels.append(idx)

            if len(self.filepaths) > 0:
                self.features_mode = True
                print(f"[ArchivalSoundtrackRNNDataset] Discovered {len(self.filepaths)} precomputed CQT+HPSS feature files in {self.features_dir} (Fast Cached Mode).")

        # 2. Dynamic waveform loading mode (if precompute_features is False or fallback)
        if not self.features_mode:
            for class_name, idx in self.class_to_idx.items():
                class_dir = os.path.join(self.data_dir, class_name)
                if os.path.isdir(class_dir):
                    files = sorted(
                        glob.glob(os.path.join(class_dir, "*.wav")) + 
                        glob.glob(os.path.join(class_dir, "*.mp3")) +
                        glob.glob(os.path.join(class_dir, "*.flac"))
                    )
                    for f in files:
                        self.filepaths.append(f)
                        self.labels.append(idx)

            print(f"[ArchivalSoundtrackRNNDataset] Loaded {len(self.filepaths)} raw audio files across {len(self.class_to_idx)} classes from {self.data_dir} (Dynamic Mode).")

    def __len__(self) -> int:
        return len(self.filepaths)

    def _pad_short_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Inserts a short event (e.g., 1–3s percussion strike) at a random position 
        within the 8s silent canvas instead of tiling.
        
        Avoids the 'Tiling Trap' where short percussion strikes are looped 5-6x,
        which erroneously trains the recurrent layer to learn rapid periodic tempo loops
        (120-180 BPM) instead of the isolated attack envelope and timbral characteristics.
        """
        if len(audio) >= self.num_samples:
            return audio

        # Low-level baseline noise floor (~ -66 dBFS) mimicking subtle optical track noise
        noise_floor = np.random.normal(0.0, 0.0005, size=self.num_samples).astype(np.float32)

        # 2. Pick a random start point so the attack happens at different times
        max_start = self.num_samples - len(audio)
        start_idx = random.randint(0, max_start) if self.random_crop else max_start // 2

        # Place transient over the ambient noise floor
        noise_floor[start_idx : start_idx + len(audio)] += audio
        return np.clip(noise_floor, -1.0, 1.0)

    # Alias for backward compatibility
    _pad_audio = _pad_short_audio

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.filepaths[idx]
        label = self.labels[idx]

        # 1. Fast loading of precomputed CQT + HPSS features (Bypasses waveform transform)
        if self.features_mode:
            features = torch.load(path, map_location="cpu", weights_only=True).float()
            target_frames = int(np.ceil(self.num_samples / self.cqt_hop))
            t_len = features.shape[-1]
            if t_len > target_frames:
                if self.random_crop:
                    start_t = random.randint(0, t_len - target_frames)
                else:
                    start_t = (t_len - target_frames) // 2
                features = features[:, :, start_t : start_t + target_frames]
            elif t_len < target_frames:
                pad_t = target_frames - t_len
                # Pad with minimum value (silence floor in log-spectrogram space) instead of replicate
                min_val = float(features.min())
                features = torch.nn.functional.pad(features, (0, pad_t), mode="constant", value=min_val)

            # Feature Standardisation (Instance z-score per channel)
            # Mitigates dynamic range shift across different historical tape transfers
            f_mean = features.mean(dim=(-2, -1), keepdim=True)
            f_std = features.std(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
            features = (features - f_mean) / f_std

            return features, label

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

        # 1. Peak Normalisation
        peak = np.max(np.abs(audio))
        if peak > 1e-4:
            audio = audio / peak

        # 2. RMS Energy Normalisation (Standardize loudness across archival transfers)
        target_rms = 0.1
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 1e-4:
            audio = audio * (target_rms / current_rms)
        audio = np.clip(audio, -1.0, 1.0)

        # Pad if shorter than target samples (Random Offset Placement)
        if len(audio) < self.num_samples:
            audio = self._pad_short_audio(audio)
        elif len(audio) > self.num_samples:
            if self.random_crop:
                start_idx = random.randint(0, len(audio) - self.num_samples)
            else:
                start_idx = (len(audio) - self.num_samples) // 2
            audio = audio[start_idx:start_idx + self.num_samples]

        # Apply synthetic degradation during training
        if self.degradation_enabled and random.random() < self.degradation_prob:
            if self.degrader is None:
                self.degrader = SyntheticDegradation(sr=self.sr, config=self.degradation_config)
            try:
                audio = self.degrader.apply(audio)
            except Exception:
                pass

        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        return audio_tensor, label
