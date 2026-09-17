import gc
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch
from tqdm import tqdm


class OpticalChannelFilter:
    """
    Simulates optical audio recording frequency constraints and analog projector electrical noise.
    - 80 Hz – 7.5 kHz Butterworth bandpass filter
    - 50 Hz optical mains hum with subtle harmonic injection
    """
    def __init__(
        self,
        sr: int,
        low_hz: float = 80.0,
        high_hz: float = 7500.0,
        order: int = 2,
        hum_freq_hz: float = 50.0,
        hum_level: float = 0.005
    ):
        self.sr = sr
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.hum_freq_hz = hum_freq_hz
        self.hum_level = hum_level

        nyquist = 0.5 * sr
        safe_high = min(high_hz, nyquist * 0.98)
        safe_low = max(10.0, low_hz)

        # 2nd order Butterworth bandpass
        self.sos = signal.butter(
            N=order,
            Wn=[safe_low, safe_high],
            btype="bandpass",
            fs=sr,
            output="sos"
        )

    def apply_filter(self, y: np.ndarray) -> np.ndarray:
        if len(y) < 16:
            return y
        return signal.sosfilt(self.sos, y).astype(np.float32)

    def inject_hum(self, y: np.ndarray) -> np.ndarray:
        num_samples = len(y)
        t = np.arange(num_samples) / float(self.sr)
        # Fundamental hum + 2nd harmonic (100 Hz / 120 Hz)
        hum = np.sin(2 * np.pi * self.hum_freq_hz * t) + 0.3 * np.sin(4 * np.pi * self.hum_freq_hz * t)
        hum = (hum * self.hum_level).astype(np.float32)
        return y + hum


class SegmentAugmentor:
    """
    Phase 2: Segment Slicing & Augmentation
    - Slices audio into aligned 4.0s windows with 1.0s hop across 24kHz and 48kHz streams
    - Generates synthetic polyphonic mixups across classes
    - Applies 80Hz–7.5kHz optical bandpass filtering and optical hum injection
    - Saves augmented tensors / manifest to disk buffer
    """
    def __init__(
        self,
        class_labels: Optional[List[str]] = None,
        window_sec: float = 4.0,
        hop_sec: float = 1.0,
        sr_mert: int = 24000,
        sr_clap: int = 48000,
        mixup_prob: float = 0.5,
        mixup_alpha_range: Tuple[float, float] = (0.3, 0.7),
        apply_optical_filter: bool = True,
        bandpass_low_hz: float = 80.0,
        bandpass_high_hz: float = 7500.0,
        hum_freq_hz: float = 50.0,
        hum_level: float = 0.005
    ):
        self.class_labels = class_labels or [
            "erhu", "pipa", "dizi", "percussion", "strings_western", "sheng",
            "suona", "guzheng", "yangqin", "woodwind", "brass", "nonmusic"
        ]
        self.class_to_idx = {c: i for i, c in enumerate(self.class_labels)}
        self.num_classes = len(self.class_labels)

        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.sr_mert = sr_mert
        self.sr_clap = sr_clap

        self.win_mert = int(window_sec * sr_mert)
        self.hop_mert = int(hop_sec * sr_mert)
        self.win_clap = int(window_sec * sr_clap)
        self.hop_clap = int(hop_sec * sr_clap)

        self.mixup_prob = mixup_prob
        self.mixup_alpha_range = mixup_alpha_range
        self.apply_optical_filter = apply_optical_filter

        self.filter_mert = OpticalChannelFilter(
            sr=sr_mert, low_hz=bandpass_low_hz, high_hz=bandpass_high_hz,
            hum_freq_hz=hum_freq_hz, hum_level=hum_level
        )
        self.filter_clap = OpticalChannelFilter(
            sr=sr_clap, low_hz=bandpass_low_hz, high_hz=bandpass_high_hz,
            hum_freq_hz=hum_freq_hz, hum_level=hum_level
        )

    def slice_file(
        self,
        path_24k: str,
        path_48k: str,
        class_name: str,
        recording_id: str
    ) -> List[Dict[str, Any]]:
        """
        Slices a paired 24kHz / 48kHz audio recording into aligned 4.0s windows with 1.0s hop.
        """
        y_24k, _ = sf.read(path_24k, dtype="float32")
        y_48k, _ = sf.read(path_48k, dtype="float32")

        # Ensure 1D
        if y_24k.ndim > 1:
            y_24k = y_24k.mean(axis=-1)
        if y_48k.ndim > 1:
            y_48k = y_48k.mean(axis=-1)

        total_sec = min(len(y_24k) / float(self.sr_mert), len(y_48k) / float(self.sr_clap))
        if total_sec < 0.5:
            return []

        segments: List[Dict[str, Any]] = []
        start_sec = 0.0

        one_hot = np.zeros(self.num_classes, dtype=np.float32)
        if class_name in self.class_to_idx:
            one_hot[self.class_to_idx[class_name]] = 1.0

        while start_sec + self.window_sec <= total_sec or start_sec == 0.0:
            # 24k segment
            s_24 = int(start_sec * self.sr_mert)
            e_24 = s_24 + self.win_mert
            seg_24 = y_24k[s_24:e_24]
            if len(seg_24) < self.win_mert:
                seg_24 = np.pad(seg_24, (0, self.win_mert - len(seg_24)), mode="reflect")

            # 48k segment
            s_48 = int(start_sec * self.sr_clap)
            e_48 = s_48 + self.win_clap
            seg_48 = y_48k[s_48:e_48]
            if len(seg_48) < self.win_clap:
                seg_48 = np.pad(seg_48, (0, self.win_clap - len(seg_48)), mode="reflect")

            segments.append({
                "audio_24k": seg_24,
                "audio_48k": seg_48,
                "labels": one_hot.copy(),
                "class": class_name,
                "recording_id": recording_id,
                "start_sec": start_sec,
                "end_sec": start_sec + self.window_sec
            })

            start_sec += self.hop_sec
            if start_sec + self.window_sec > total_sec:
                break

        return segments

    def augment_segments(
        self,
        segments: List[Dict[str, Any]],
        num_mixups: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Applies synthetic polyphonic mixups and optical bandpass + hum degradation.
        """
        augmented: List[Dict[str, Any]] = []

        # 1. Optical filtering and hum on base slices
        for seg in segments:
            a_24 = seg["audio_24k"]
            a_48 = seg["audio_48k"]

            if self.apply_optical_filter:
                a_24 = self.filter_mert.inject_hum(self.filter_mert.apply_filter(a_24))
                a_48 = self.filter_clap.inject_hum(self.filter_clap.apply_filter(a_48))

            augmented.append({
                "audio_24k": a_24,
                "audio_48k": a_48,
                "labels": seg["labels"],
                "recording_id": seg["recording_id"],
                "class": seg["class"],
                "is_mixup": False
            })

        # 2. Synthetic polyphonic mixup generation
        if self.mixup_prob > 0 and len(segments) > 1:
            total_mixups = num_mixups or int(len(segments) * self.mixup_prob)
            print(f"Generating {total_mixups} synthetic polyphonic mixup crops...")

            for _ in range(total_mixups):
                idx_a = random.randrange(len(segments))
                idx_b = random.randrange(len(segments))
                seg_a = segments[idx_a]
                seg_b = segments[idx_b]

                # Try to mix different classes if possible
                if seg_a["class"] == seg_b["class"] and len(segments) > 10:
                    idx_b = (idx_b + random.randint(1, len(segments) - 1)) % len(segments)
                    seg_b = segments[idx_b]

                alpha = random.uniform(self.mixup_alpha_range[0], self.mixup_alpha_range[1])
                mix_24 = alpha * seg_a["audio_24k"] + (1.0 - alpha) * seg_b["audio_24k"]
                mix_48 = alpha * seg_a["audio_48k"] + (1.0 - alpha) * seg_b["audio_48k"]

                # Multi-hot union label
                mix_labels = np.maximum(seg_a["labels"], seg_b["labels"])

                if self.apply_optical_filter:
                    mix_24 = self.filter_mert.inject_hum(self.filter_mert.apply_filter(mix_24))
                    mix_48 = self.filter_clap.inject_hum(self.filter_clap.apply_filter(mix_48))

                augmented.append({
                    "audio_24k": mix_24,
                    "audio_48k": mix_48,
                    "labels": mix_labels,
                    "recording_id": f"{seg_a['recording_id']}_mix_{seg_b['recording_id']}",
                    "class": f"{seg_a['class']}+{seg_b['class']}",
                    "is_mixup": True
                })

        return augmented

    def load_slice(
        self,
        path_24k: str,
        path_48k: str,
        start_sec: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads a single 4.0s window on-demand directly from standardized WAV files.
        """
        start_frame_24 = int(round(start_sec * self.sr_mert))
        frames_24 = self.win_mert
        with sf.SoundFile(path_24k) as f24:
            f24.seek(start_frame_24)
            c_24 = f24.read(frames_24, dtype="float32")
        if len(c_24) < frames_24:
            c_24 = np.pad(c_24, (0, frames_24 - len(c_24)), mode="reflect")
        if c_24.ndim > 1:
            c_24 = c_24.mean(axis=-1)

        start_frame_48 = int(round(start_sec * self.sr_clap))
        frames_48 = self.win_clap
        with sf.SoundFile(path_48k) as f48:
            f48.seek(start_frame_48)
            c_48 = f48.read(frames_48, dtype="float32")
        if len(c_48) < frames_48:
            c_48 = np.pad(c_48, (0, frames_48 - len(c_48)), mode="reflect")
        if c_48.ndim > 1:
            c_48 = c_48.mean(axis=-1)

        return c_24, c_48

    def process_standardized_directory(
        self,
        standardized_dir: str = "data/ast/standardized",
        output_buffer_dir: str = "data/ast/augmented_segments",
        shard_size: int = 500
    ) -> str:
        """
        Slices all standardized recordings and serializes augmented tensors to disk buffer
        using memory-efficient sharding (capped to ~500 MB RAM to prevent macOS OOM Killed: 9).
        """
        out_path = Path(output_buffer_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        # Clean existing shards if any
        for old_shard in out_path.glob("shard_*.pt"):
            try:
                old_shard.unlink()
            except Exception:
                pass

        std_path = Path(standardized_dir)
        dir_24k = std_path / "24k"
        dir_48k = std_path / "48k"

        if not dir_24k.exists() or not dir_48k.exists():
            raise FileNotFoundError(f"Standardized directories not found: {dir_24k}, {dir_48k}")

        files_24k = sorted([p for p in dir_24k.rglob("*.wav") if not p.name.startswith(".")])
        print(f"Phase 2: Slicing {len(files_24k)} standardized files (4.0s window, {self.hop_sec}s hop)...")

        # Step 1: Lightweight slice metadata indexing (RAM < 10 MB)
        slice_meta: List[Dict[str, Any]] = []
        for f_24 in files_24k:
            rel = f_24.relative_to(dir_24k)
            f_48 = dir_48k / rel
            if not f_48.exists():
                continue

            class_name = rel.parent.name
            recording_id = rel.stem
            info = sf.info(str(f_24))
            total_sec = info.duration

            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            if class_name in self.class_to_idx:
                one_hot[self.class_to_idx[class_name]] = 1.0

            start_sec = 0.0
            while start_sec + self.window_sec <= total_sec or start_sec == 0.0:
                slice_meta.append({
                    "path_24k": str(f_24),
                    "path_48k": str(f_48),
                    "start_sec": start_sec,
                    "class": class_name,
                    "recording_id": recording_id,
                    "labels": one_hot.copy()
                })
                start_sec += self.hop_sec
                if start_sec + self.window_sec > total_sec:
                    break

        print(f"Base windows indexed: {len(slice_meta)}")

        # Step 2: Stream base slices to sharded disk buffer
        shard_idx = 0
        current_shard: List[Dict[str, Any]] = []
        shard_paths: List[str] = []
        total_created = 0

        def save_current_shard():
            nonlocal shard_idx, current_shard, shard_paths
            if not current_shard:
                return
            s_path = out_path / f"shard_{shard_idx:04d}.pt"
            torch.save(current_shard, str(s_path))
            shard_paths.append(str(s_path))
            shard_idx += 1
            current_shard.clear()
            gc.collect()

        print(f"Streaming base slices to disk buffer in shards of {shard_size}...")
        for item in tqdm(slice_meta, desc="Buffering base slices"):
            c_24, c_48 = self.load_slice(item["path_24k"], item["path_48k"], item["start_sec"])
            if self.apply_optical_filter:
                c_24 = self.filter_mert.inject_hum(self.filter_mert.apply_filter(c_24))
                c_48 = self.filter_clap.inject_hum(self.filter_clap.apply_filter(c_48))

            current_shard.append({
                "audio_24k": c_24,
                "audio_48k": c_48,
                "labels": item["labels"],
                "recording_id": item["recording_id"],
                "class": item["class"],
                "is_mixup": False
            })
            total_created += 1

            if len(current_shard) >= shard_size:
                save_current_shard()

        # Step 3: Stream synthetic polyphonic mixups
        if self.mixup_prob > 0 and len(slice_meta) > 1:
            total_mixups = int(len(slice_meta) * self.mixup_prob)
            print(f"Generating and streaming {total_mixups} synthetic polyphonic mixup crops...")

            for _ in tqdm(range(total_mixups), desc="Buffering mixups"):
                idx_a = random.randrange(len(slice_meta))
                idx_b = random.randrange(len(slice_meta))
                meta_a = slice_meta[idx_a]
                meta_b = slice_meta[idx_b]

                if meta_a["class"] == meta_b["class"] and len(slice_meta) > 10:
                    idx_b = (idx_b + random.randint(1, len(slice_meta) - 1)) % len(slice_meta)
                    meta_b = slice_meta[idx_b]

                a_24, a_48 = self.load_slice(meta_a["path_24k"], meta_a["path_48k"], meta_a["start_sec"])
                b_24, b_48 = self.load_slice(meta_b["path_24k"], meta_b["path_48k"], meta_b["start_sec"])

                alpha = random.uniform(self.mixup_alpha_range[0], self.mixup_alpha_range[1])
                mix_24 = alpha * a_24 + (1.0 - alpha) * b_24
                mix_48 = alpha * a_48 + (1.0 - alpha) * b_48
                mix_labels = np.maximum(meta_a["labels"], meta_b["labels"])

                if self.apply_optical_filter:
                    mix_24 = self.filter_mert.inject_hum(self.filter_mert.apply_filter(mix_24))
                    mix_48 = self.filter_clap.inject_hum(self.filter_clap.apply_filter(mix_48))

                current_shard.append({
                    "audio_24k": mix_24,
                    "audio_48k": mix_48,
                    "labels": mix_labels,
                    "recording_id": f"{meta_a['recording_id']}_mix_{meta_b['recording_id']}",
                    "class": f"{meta_a['class']}+{meta_b['class']}",
                    "is_mixup": True
                })
                total_created += 1

                if len(current_shard) >= shard_size:
                    save_current_shard()

        # Flush remaining segments to the final shard
        save_current_shard()

        manifest = {
            "shards": shard_paths,
            "total_segments": total_created,
            "num_classes": self.num_classes,
            "class_labels": self.class_labels
        }
        manifest_path = out_path / "segments_manifest.pt"
        torch.save(manifest, str(manifest_path))
        print(f"Phase 2 complete: {total_created} segments buffered across {len(shard_paths)} shards.")
        print(f"Manifest saved to: {manifest_path}")
        return str(manifest_path)
