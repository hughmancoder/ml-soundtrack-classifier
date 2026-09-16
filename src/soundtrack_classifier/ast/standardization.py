import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm


class AudioStandardizer:
    """
    Phase 1: Audio Standardization
    Standardizes raw dataset recordings into dual-stream 32-bit float mono WAV files.
    - Downmixes multi-channel stereo audio to mono
    - Trims leading/trailing silence with 50ms pre-roll and 2.0s post-roll
    - Dual-stream resampling via SoXR high-quality backend:
        - 24 kHz stream for MERT-v1-95M
        - 48 kHz stream for LAION-CLAP
    """
    def __init__(
        self,
        sr_mert: int = 24000,
        sr_clap: int = 48000,
        pre_roll_sec: float = 0.05,
        post_roll_sec: float = 2.0,
        silence_top_db: float = 40.0,
        res_type: str = "soxr_vhq"
    ):
        self.sr_mert = sr_mert
        self.sr_clap = sr_clap
        self.pre_roll_sec = pre_roll_sec
        self.post_roll_sec = post_roll_sec
        self.silence_top_db = silence_top_db
        self.res_type = res_type

    def standardize_file(
        self,
        input_path: str,
        output_dir_24k: str,
        output_dir_48k: str,
        class_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes a single audio file and outputs 24kHz and 48kHz WAV files.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # 1. Load audio maintaining native sampling rate and full dynamic range
        y, orig_sr = librosa.load(str(input_path), sr=None, mono=True)
        if y.ndim > 1:
            y = librosa.to_mono(y)

        # 2. Trim silence with pre-roll and post-roll
        if len(y) > 0 and self.silence_top_db is not None:
            _, index = librosa.effects.trim(y, top_db=self.silence_top_db)
            start_idx, end_idx = index[0], index[1]
            pre_roll = int(self.pre_roll_sec * orig_sr)
            post_roll = int(self.post_roll_sec * orig_sr)
            start = max(0, start_idx - pre_roll)
            end = min(len(y), end_idx + post_roll)
            y_trimmed = y[start:end]
            if len(y_trimmed) == 0:
                y_trimmed = y
        else:
            y_trimmed = y

        # 3. Resample dual streams with SoXR
        # 24 kHz for MERT
        if orig_sr != self.sr_mert:
            y_24k = librosa.resample(y_trimmed, orig_sr=orig_sr, target_sr=self.sr_mert, res_type=self.res_type)
        else:
            y_24k = y_trimmed.copy()

        # 48 kHz for CLAP
        if orig_sr != self.sr_clap:
            y_48k = librosa.resample(y_trimmed, orig_sr=orig_sr, target_sr=self.sr_clap, res_type=self.res_type)
        else:
            y_48k = y_trimmed.copy()

        # 4. Save standardized 32-bit float WAV files
        cls_subdir = class_name or input_path.parent.name
        stem_name = input_path.stem

        out_24k_dir = Path(output_dir_24k) / cls_subdir
        out_48k_dir = Path(output_dir_48k) / cls_subdir
        out_24k_dir.mkdir(parents=True, exist_ok=True)
        out_48k_dir.mkdir(parents=True, exist_ok=True)

        out_path_24k = out_24k_dir / f"{stem_name}.wav"
        out_path_48k = out_48k_dir / f"{stem_name}.wav"

        sf.write(str(out_path_24k), y_24k.astype(np.float32), self.sr_mert, subtype="FLOAT")
        sf.write(str(out_path_48k), y_48k.astype(np.float32), self.sr_clap, subtype="FLOAT")

        return {
            "id": stem_name,
            "class": cls_subdir,
            "path_24k": str(out_path_24k),
            "path_48k": str(out_path_48k),
            "duration_sec": len(y_24k) / float(self.sr_mert),
            "num_samples_24k": len(y_24k),
            "num_samples_48k": len(y_48k)
        }

    def process_directory(
        self,
        data_dir: str,
        output_dir: str = "data/ast/standardized",
        audio_extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    ) -> List[Dict[str, Any]]:
        """
        Discovers all audio files across class subdirectories and runs standardization.
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        out_24k = output_path / "24k"
        out_48k = output_path / "48k"

        files = [
            p for p in data_path.rglob("*")
            if p.suffix.lower() in audio_extensions and not p.name.startswith(".")
        ]

        print(f"Phase 1: Standardizing {len(files)} raw audio files from {data_dir}...")
        manifest: List[Dict[str, Any]] = []

        for f in tqdm(files, desc="Standardizing audio (24kHz & 48kHz)"):
            try:
                item = self.standardize_file(
                    input_path=str(f),
                    output_dir_24k=str(out_24k),
                    output_dir_48k=str(out_48k),
                    class_name=f.parent.name
                )
                manifest.append(item)
            except Exception as e:
                print(f"Warning: Failed to standardize {f}: {e}")

        print(f"Phase 1 Complete: {len(manifest)} files standardized into {output_dir}/")
        return manifest
