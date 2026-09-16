#!/usr/bin/env python3
"""
Soundtrack Classifier CNN Inference CLI Pipeline.
Runs sliding window inference across continuous audio using the fine-tuned CNN model.

Usage:
    make cnn-pipeline input="data/test/sample.wav"
    # or directly:
    PYTHONPATH=src python scripts/cnn_pipeline.py "data/test/sample.wav"
"""

import os
import sys
import argparse
import yaml
import torch
import librosa
import numpy as np
from pathlib import Path
from nnAudio.features.cqt import CQT1992v2
from scipy.signal import medfilt

from soundtrack_classifier.cnn.resnet import ResNet50Classifier, ResNet18Classifier


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_inference(input_file: str, model_path: str = None):
    # 1. Configuration & Device Setup
    config = load_config()
    cnn_cfg = config.get("cnn", {})
    audio_cfg = cnn_cfg.get("audio", config.get("audio", {}))
    paths_cfg = cnn_cfg.get("paths", config.get("paths", {}))
    
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Running inference on device: {device}")

    # Determine default model path
    if model_path is None:
        models_dir = paths_cfg.get("models_dir", "models")
        candidates = [
            os.path.join(models_dir, "resnet50_cnn.pt"),
            os.path.join(models_dir, "resnet50_cnn_best.pt"),
            os.path.join(models_dir, "resnet18_cnn_best.pt"),
            os.path.join(models_dir, "resnet18_cnn.pt")
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break
        if model_path is None:
            model_path = os.path.join(models_dir, "resnet50_cnn.pt")

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}. Please train a model first (make train-cnn).")
        return

    # 2. Load Checkpoint & Model
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    labels = config.get("train_labels", [
        "erhu", "pipa", "dizi", "percussion", "strings_western", "sheng",
        "suona", "guzheng", "yangqin", "woodwind", "brass", "nonmusic"
    ])
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
        if "classes" in checkpoint:
            labels = checkpoint["classes"]
            print(f"Loaded {len(labels)} classes from checkpoint metadata.")
    else:
        state_dict = checkpoint

    idx_to_class = {i: cls_name for i, cls_name in enumerate(labels)}

    try:
        model = ResNet50Classifier(num_classes=len(labels)).to(device)
        model.load_state_dict(state_dict)
        print("Loaded ResNet-50 architecture successfully.")
    except Exception:
        print("Detected legacy ResNet-18 weights in checkpoint. Loading ResNet18Classifier...")
        model = ResNet18Classifier(num_classes=len(labels)).to(device)
        model.load_state_dict(state_dict)

    model.eval()

    # 3. Audio & CQT Setup
    sr = int(audio_cfg.get("sample_rate", 44100))
    duration = float(audio_cfg.get("duration_sec", 4.0))
    window_samples = int(duration * sr)
    stride_samples = int(1.0 * sr) # 1 second sliding stride

    cqt_cfg = audio_cfg.get("cqt", {})
    fmin = float(cqt_cfg.get("fmin", 65.4))
    n_bins = int(cqt_cfg.get("n_bins", 84))
    bins_per_octave = int(cqt_cfg.get("bins_per_octave", 12))

    cqt_extractor = CQT1992v2(
        sr=sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave
    ).to(device)

    # 4. Load Audio
    print(f"Loading audio track: {input_file}...")
    audio, _ = librosa.load(input_file, sr=sr, mono=True)
    num_samples = len(audio)
    
    if num_samples < window_samples:
        print(f"Audio duration ({num_samples/sr:.2f}s) is shorter than window length ({duration}s). Padding...")
        audio = np.pad(audio, (0, window_samples - num_samples))
        num_samples = len(audio)

    raw_predictions = []
    timestamps_sec = []

    print("Running sliding window inference...")
    with torch.no_grad():
        for start_idx in range(0, num_samples - window_samples + 1, stride_samples):
            end_idx = start_idx + window_samples
            chunk = audio[start_idx:end_idx]

            # RMS silence cutoff
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < 0.002:
                # Silence frame
                continue

            audio_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            cqt_spec = cqt_extractor(audio_tensor)
            cqt_spec = cqt_spec.unsqueeze(1) # [1, 1, Freq, Time]

            logits = model(cqt_spec)
            pred_idx = torch.argmax(logits, dim=-1).item()

            raw_predictions.append(pred_idx)
            timestamps_sec.append(start_idx / sr)

    if not raw_predictions:
        print("No acoustic activity detected above silence threshold.")
        return

    # 5. Temporal Filtering & Contiguous Merging
    print("Applying smoothing and event aggregation...")
    smoothed_preds = medfilt(raw_predictions, kernel_size=3).astype(int)

    merged_events = []
    current_label = smoothed_preds[0]
    start_time = timestamps_sec[0]

    for i in range(1, len(smoothed_preds)):
        if smoothed_preds[i] != current_label:
            end_time = timestamps_sec[i] + (duration - 1.0)
            if (end_time - start_time) >= 0.5:
                merged_events.append({
                    "instrument": idx_to_class[current_label],
                    "start": start_time,
                    "end": end_time
                })
            current_label = smoothed_preds[i]
            start_time = timestamps_sec[i]

    end_time = timestamps_sec[-1] + duration
    merged_events.append({
        "instrument": idx_to_class[current_label],
        "start": start_time,
        "end": end_time
    })

    # 6. Output Summary
    print("\n--- Final Extracted Acoustic Events ---")
    for event in merged_events:
        print(f"[{event['start']:06.1f}s - {event['end']:06.1f}s] : {event['instrument']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sliding window inference using Soundtrack Classifier CNN")
    parser.add_argument("input_file", type=str, help="Path to the audio file (.wav)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained CNN model weights")
    args = parser.parse_args()

    run_inference(args.input_file, args.model)
