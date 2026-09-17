#!/usr/bin/env python3
"""
Debug Samples Generator for Archival CRNN Pipeline.

Extracts a small representative set of clean training audio clips,
applies the full optical sound channel degradation pipeline (5.5 kHz Butterworth LP,
100 Hz HP, non-linear tanh saturation, optical film hiss & projector hum at 12-20 dB SNR),
and saves:
  1. Clean 8s WAV audio
  2. Degraded 8s WAV audio
  3. Side-by-side 2-Channel HPSS CQT spectrogram comparison plots (Harmonic & Percussive)

Outputs are saved to data/debug/ by default.

Usage:
    PYTHONPATH=src python scripts/debug_samples.py
    # Or via make:
    make debug
"""

import os
import sys
import glob
import random
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

# Ensure src/ is in sys.path
repo_root = Path(__file__).resolve().parent.parent
src_dir = str(repo_root / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from soundtrack_classifier.ingestion.degradation import SyntheticDegradation
from soundtrack_classifier.features.cqt import CQTExtractor


def load_config(config_path="config/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def pad_or_crop_audio(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Standardises waveform to target length using ambient noise floor padding for short
    clips and center crop for long clips.
    """
    if len(audio) < target_samples:
        # Low-level optical baseline noise floor (~ -66 dBFS)
        noise_floor = np.random.normal(0.0, 0.0005, size=target_samples).astype(np.float32)
        start_idx = (target_samples - len(audio)) // 2
        noise_floor[start_idx : start_idx + len(audio)] += audio
        return np.clip(noise_floor, -1.0, 1.0)
    elif len(audio) > target_samples:
        start_idx = (len(audio) - target_samples) // 2
        return audio[start_idx : start_idx + target_samples]
    return audio


def main():
    parser = argparse.ArgumentParser(
        description="Generate clean vs degraded audio samples and CQT comparisons for the CRNN pipeline."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/debug",
        help="Directory to save clean/degraded WAVs and comparison plots (default: data/debug)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of diverse sample files to process (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config()
    crnn_cfg = config.get("crnn", {})
    audio_cfg = crnn_cfg.get("audio", {})
    cqt_cfg = audio_cfg.get("cqt", {})
    degradation_cfg = crnn_cfg.get("degradation", {})

    sr = int(audio_cfg.get("sample_rate", 22050))
    duration_sec = float(audio_cfg.get("crop_duration_sec", 8.0))
    target_samples = int(duration_sec * sr)
    data_dir = (
        config.get("paths", {}).get("data_dir") or
        crnn_cfg.get("paths", {}).get("data_dir") or
        "data/train"
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Initializing Archival CRNN Debug Samples Generator:")
    print(f"    Source data dir : {data_dir}")
    print(f"    Target debug dir: {output_dir}")
    print(f"    Sample rate     : {sr} Hz")
    print(f"    Duration        : {duration_sec}s ({target_samples} samples)")

    # 1. Initialize optical sound channel degrader
    degrader = SyntheticDegradation(sr=sr, config=degradation_cfg)

    # 2. Initialize 2-channel CQT + HPSS extractor
    device_str = "cpu"  # Keep on CPU for reliable matplotlib visualization & median filter
    cqt_extractor = CQTExtractor(
        sr=sr,
        fmin=float(cqt_cfg.get("fmin", 65.4)),
        n_bins=int(cqt_cfg.get("n_bins", 168)),
        bins_per_octave=int(cqt_cfg.get("bins_per_octave", 24)),
        hop_length=int(cqt_cfg.get("hop_length", 512)),
        device=device_str
    )

    # 3. Select diverse sample files across different classes
    class_folders = [
        d for d in glob.glob(os.path.join(data_dir, "*"))
        if os.path.isdir(d) and not os.path.basename(d).startswith(".")
    ]
    if not class_folders:
        print(f"Error: No class folders found in {data_dir}.")
        sys.exit(1)

    # Gather candidate files per class
    available_classes = {}
    for c_dir in sorted(class_folders):
        c_name = os.path.basename(c_dir)
        files = (
            glob.glob(os.path.join(c_dir, "*.wav")) +
            glob.glob(os.path.join(c_dir, "*.mp3")) +
            glob.glob(os.path.join(c_dir, "*.flac"))
        )
        if files:
            available_classes[c_name] = files

    if not available_classes:
        print(f"Error: No audio files found in {data_dir}.")
        sys.exit(1)

    # Pick samples from distinct classes (preferring diverse types: wind, plucked, percussion)
    chosen_classes = list(available_classes.keys())
    random.shuffle(chosen_classes)
    chosen_classes = chosen_classes[: min(args.num_samples, len(chosen_classes))]

    sample_paths = [random.choice(available_classes[c]) for c in chosen_classes]

    print(f"\nProcessing {len(sample_paths)} diverse samples...")
    results = []

    for i, file_path in enumerate(sample_paths, 1):
        instrument = Path(file_path).parent.name
        base_name = Path(file_path).stem
        clean_base = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_name)
        prefix = f"{instrument}_{clean_base}"

        print(f"\n[{i}/{len(sample_paths)}] Processing: {instrument} ({Path(file_path).name})")

        # A. Load audio preserving original signal and resample to 22050 Hz
        try:
            audio, f_sr = sf.read(file_path, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            if f_sr != sr:
                audio = librosa.resample(audio, orig_sr=f_sr, target_sr=sr)
        except Exception:
            audio, _ = librosa.load(file_path, sr=sr, mono=True)

        # B. Standard Peak & Target RMS Normalisation
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-4:
            audio = audio / peak

        target_rms = 0.1
        current_rms = float(np.sqrt(np.mean(audio ** 2)))
        if current_rms > 1e-4:
            audio = audio * (target_rms / current_rms)
        audio = np.clip(audio, -1.0, 1.0)

        # C. Pad or crop to exact 8.0s canvas
        clean_audio = pad_or_crop_audio(audio, target_samples)

        # D. Apply Synthetic Archival Degradation
        degraded_audio = degrader.apply(clean_audio)

        # E. Save audio WAV files
        clean_wav_path = os.path.join(output_dir, f"{prefix}_clean.wav")
        degraded_wav_path = os.path.join(output_dir, f"{prefix}_degraded.wav")
        sf.write(clean_wav_path, clean_audio, sr)
        sf.write(degraded_wav_path, degraded_audio, sr)
        print(f"  -> Saved clean WAV    : {clean_wav_path}")
        print(f"  -> Saved degraded WAV : {degraded_wav_path}")

        # F. Compute 2-Channel HPSS CQT Features
        clean_tensor = torch.tensor(clean_audio, dtype=torch.float32)
        degraded_tensor = torch.tensor(degraded_audio, dtype=torch.float32)

        with torch.no_grad():
            clean_feat = cqt_extractor.extract(clean_tensor, return_hpss=True).squeeze(0).cpu().numpy()
            degraded_feat = cqt_extractor.extract(degraded_tensor, return_hpss=True).squeeze(0).cpu().numpy()

        # Log-magnitude compression matching model front-end
        clean_h = np.log1p(10.0 * clean_feat[0])
        clean_p = np.log1p(10.0 * clean_feat[1])
        deg_h = np.log1p(10.0 * degraded_feat[0])
        deg_p = np.log1p(10.0 * degraded_feat[1])

        # G. Plot 4-Panel Side-by-Side Comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)

        # Harmonic Channel (Channel 0)
        im0 = axes[0, 0].imshow(clean_h, aspect="auto", origin="lower", cmap="magma")
        axes[0, 0].set_title(f"Clean Harmonic CQT [{instrument}]", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("CQT Frequency Bins (65 Hz - 8 kHz)")
        fig.colorbar(im0, ax=axes[0, 0], format="%.1f")

        im1 = axes[0, 1].imshow(deg_h, aspect="auto", origin="lower", cmap="magma")
        axes[0, 1].set_title(f"Degraded Harmonic CQT (5.5kHz Rolloff + Tanh)", fontsize=12, fontweight="bold")
        fig.colorbar(im1, ax=axes[0, 1], format="%.1f")

        # Percussive Channel (Channel 1)
        im2 = axes[1, 0].imshow(clean_p, aspect="auto", origin="lower", cmap="viridis")
        axes[1, 0].set_title(f"Clean Percussive CQT [{instrument}]", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Time Frames (Hop = 512 / ~23ms)")
        axes[1, 0].set_ylabel("CQT Frequency Bins (65 Hz - 8 kHz)")
        fig.colorbar(im2, ax=axes[1, 0], format="%.1f")

        im3 = axes[1, 1].imshow(deg_p, aspect="auto", origin="lower", cmap="viridis")
        axes[1, 1].set_title(f"Degraded Percussive CQT (Optical Noise + Hum)", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Time Frames (Hop = 512 / ~23ms)")
        fig.colorbar(im3, ax=axes[1, 1], format="%.1f")

        snr_min = float(degradation_cfg.get("snr_min_db", 42.0))
        snr_max = float(degradation_cfg.get("snr_max_db", 52.0))
        lp_cut = float(degradation_cfg.get("lowpass_cutoff_hz", 5500.0))
        plt.suptitle(
            f"Archival Degradation Comparison: {instrument} ({clean_base[:25]})\n"
            f"SR: {sr} Hz | Duration: {duration_sec}s | LP Rolloff: {lp_cut/1000:.1f} kHz | Optical SNR: {snr_min:.0f}-{snr_max:.0f} dB (Subtle)",
            fontsize=13
        )
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{prefix}_comparison.png")
        plt.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  -> Saved CQT plot     : {plot_path}")

        results.append({
            "instrument": instrument,
            "clean_wav": clean_wav_path,
            "degraded_wav": degraded_wav_path,
            "plot": plot_path
        })

    print("\n" + "=" * 65)
    print(f"Successfully generated {len(results)} debug samples in '{output_dir}/'")
    print("=" * 65)
    for r in results:
        print(f"  - [{r['instrument']:<12}] Plot: {r['plot']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
