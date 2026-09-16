#!/usr/bin/env python3
"""
Precompute & Cache 2-Channel CQT+HPSS Features for Archival CRNN.

Pre-extracts 2-channel log-CQT+HPSS features (Harmonic & Percussive) for all training
audio files and saves them as compressed float16 .pt tensors.
During training, loading precomputed features bypasses expensive on-the-fly STFT/CQT
computations, accelerating epochs by 5x to 10x.

Usage:
    PYTHONPATH=src python scripts/precompute_cqt_hpss.py
    # Or via make:
    make precompute-features
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import yaml
import torch
import soundfile as sf
import librosa
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Ensure src/ is in sys.path
repo_root = Path(__file__).resolve().parent.parent
src_dir = str(repo_root / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from soundtrack_classifier.rnn.rnn import ArchivalInstrumentCRNN


def load_config(config_path="config/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def precompute_features(
    data_dir: str = "data/train",
    out_dir: str = "data/features_cqt_hpss",
    sr: int = 22050,
    use_half_precision: bool = True,
    force: bool = False
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Precomputing CQT+HPSS features on device: {device}")
    print(f"    Source Data Dir: {data_dir}")
    print(f"    Target Cache   : {out_dir}")
    print(f"    Sample Rate    : {sr} Hz")
    print(f"    Precision      : {'float16 (half)' if use_half_precision else 'float32'}")

    # Initialize CRNN feature extractor in eval mode
    crnn = ArchivalInstrumentCRNN(in_channels=2, sr=sr).to(device)
    crnn.eval()

    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    classes = [
        d for d in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ]

    total_saved = 0
    total_skipped = 0

    for cls in classes:
        cls_in_dir = os.path.join(data_dir, cls)
        cls_out_dir = os.path.join(out_dir, cls)
        os.makedirs(cls_out_dir, exist_ok=True)

        audio_files = (
            glob.glob(os.path.join(cls_in_dir, "*.wav")) +
            glob.glob(os.path.join(cls_in_dir, "*.mp3")) +
            glob.glob(os.path.join(cls_in_dir, "*.flac"))
        )
        if not audio_files:
            continue

        print(f"\nProcessing {len(audio_files)} files for class: {cls}")

        for fpath in tqdm(audio_files, desc=f"{cls:<12}"):
            fname = os.path.splitext(os.path.basename(fpath))[0] + ".pt"
            out_path = os.path.join(cls_out_dir, fname)
            if os.path.exists(out_path) and not force:
                total_skipped += 1
                continue

            try:
                # Load audio
                audio, file_sr = sf.read(fpath, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=-1)
                if file_sr != sr:
                    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

                # Amplitude Normalisation: Peak + Target RMS
                peak = float(np.max(np.abs(audio)))
                if peak > 1e-4:
                    audio = audio / peak

                target_rms = 0.1
                current_rms = float(np.sqrt(np.mean(audio ** 2)))
                if current_rms > 1e-4:
                    audio = audio * (target_rms / current_rms)
                audio = np.clip(audio, -1.0, 1.0)

                waveform = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)

                # Process long audio files in chunks (e.g. 30 seconds) to prevent GPU OOM or truncation
                chunk_samples = 30 * sr
                features_list = []
                
                with torch.no_grad():
                    for i in range(0, waveform.shape[1], chunk_samples):
                        chunk = waveform[:, i : i + chunk_samples]
                        # Extract 2-channel log-CQT+HPSS feature map: (1, 2, n_bins, time_frames)
                        feat_chunk = crnn.extract_features(chunk).squeeze(0).cpu()
                        features_list.append(feat_chunk)
                        
                # Concatenate all temporal chunks along the time frame axis
                features = torch.cat(features_list, dim=-1)

                if use_half_precision:
                    features = features.half()

                torch.save(features, out_path)
                total_saved += 1
            except Exception as e:
                print(f"Warning: Failed to process {fpath}: {e}")

    print(f"\n{'='*60}")
    print(f"Precompute complete: {total_saved} features newly saved, {total_skipped} skipped.")
    print(f"Cache location: {out_dir}")
    print(f"{'='*60}\n")


def main():
    config = load_config()
    crnn_cfg = config.get("crnn", {})
    audio_cfg = crnn_cfg.get("audio", {})
    train_cfg = crnn_cfg.get("training", {})

    parser = argparse.ArgumentParser(description="Precompute 2-Channel CQT+HPSS Features for CRNN")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=config.get("paths", {}).get("data_dir") or "data/train",
        help="Source audio data directory (default: data/train)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=train_cfg.get("features_dir", "data/features_cqt_hpss"),
        help="Target directory for precomputed .pt feature tensors (default: data/features_cqt_hpss)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=int(audio_cfg.get("sample_rate", 22050)),
        help="Sample rate (default: 22050)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if .pt file exists"
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Save as float32 instead of float16 (half precision)"
    )
    args = parser.parse_args()

    precompute_features(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        sr=args.sr,
        use_half_precision=not args.no_half,
        force=args.force
    )


if __name__ == "__main__":
    main()
